/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file nixl_service.h
 * @brief nixlServiceAgent — compression, staging, and encryption services
 *        layered transparently on top of nixlAgent.
 */
#ifndef NIXL_SERVICE_H
#define NIXL_SERVICE_H

#include "nixl.h"
#include "nixl_service_types.h"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

// Pimpl — complete type only needed in nixl_service.cpp
class nixlServiceAgentData;

// ---------------------------------------------------------------------------
// nixlService free functions
// ---------------------------------------------------------------------------

/**
 * @namespace nixlService
 * @brief  Free utility functions for the service layer.
 */
namespace nixlService {

/**
 * @brief  Query marshal backend requirements without an agent instance.
 *         Instantiates each compiled-in backend and calls its virtual
 *         queryRequirements() method, aggregating the results.
 *
 * Slot sizes are performance recommendations — the chunk size at which each
 * backend reaches peak throughput on the detected GPU — not worst-case output
 * bounds.  If the protocol uses double buffering, reported sizes already
 * include the 2× factor.
 *
 * @param  reqs  [out]  Populated with per-algorithm ideal slot sizes.
 * @return nixl_status_t  NIXL_SUCCESS, or error if any backend query fails
 *                        (e.g. no CUDA device present for nvCOMP).
 */
nixl_status_t marshalQuery(MarshalRequirements& reqs);

} // namespace nixlService

// ---------------------------------------------------------------------------
// nixlServiceAgent
// ---------------------------------------------------------------------------

/**
 * @class nixlServiceAgent
 * @brief Extends nixlAgent with transparent compression, staging, and encryption.
 *
 * IS-A nixlAgent: replacing `nixlAgent` with `nixlServiceAgent` at the
 * construction site is sufficient to gain service capabilities — no other
 * changes are needed in Python.  In C++, the handle type changes from
 * nixlXferReqH* to nixlServiceXferReqH*; the `= delete`d base-type overloads
 * produce a clear compile error if the old type is still used.
 *
 * DIRECT mode is identical to the base class, so users who want both plain
 * and service transfers can set mode per-transfer without switching types.
 *
 * Progress is driven by notification callbacks registered during construction
 * (_NIXLS_* prefix).  No background thread is needed for correctness; Phase 4
 * adds per-backend threads for high-concurrency performance.
 *
 * Service progress is driven automatically from getXferStatus() and getNotifs();
 * callers may use the same polling pattern as nixlAgent without any additional
 * API.  The internal driveService() method is the canonical progress driver
 * called by both; in Phase 4 it becomes the background thread body.
 */
class nixlServiceAgent : public nixlAgent {
public:
    /**
     * @brief Construct a service agent.
     *
     * Injects _NIXLS_* notification callbacks into the config before the
     * base nixlAgent constructor runs, so that all service protocol messages
     * are intercepted immediately when they arrive.
     *
     * @param name  Agent name (same semantics as nixlAgent)
     * @param cfg   Service configuration (defaultMode, optional defaultCompAlg).
     */
    nixlServiceAgent(const std::string& name, nixlServiceAgentConfig cfg);

    /**
     * @brief Destructor.  Virtual via nixlAgent.
     */
    ~nixlServiceAgent() override;

    // Non-copyable, non-movable (same constraint as nixlAgent)
    nixlServiceAgent(const nixlServiceAgent&)            = delete;
    nixlServiceAgent& operator=(const nixlServiceAgent&) = delete;
    nixlServiceAgent(nixlServiceAgent&&)                 = delete;
    nixlServiceAgent& operator=(nixlServiceAgent&&)      = delete;

    // -----------------------------------------------------------------------
    // Metadata exchange — service extensions
    // -----------------------------------------------------------------------

    /**
     * @brief  Return local metadata blob, appending service staging pool info.
     *         Must be called after registerServiceMem() and re-called after
     *         any register/deregister. Shadows nixlAgent::getLocalMD.
     */
    nixl_status_t getLocalMD(nixl_blob_t& md) const;

    /**
     * @brief  Load remote metadata, parsing service pool layout if present.
     *         Populates remoteStagingPools[agent_name] so that wire messages
     *         carrying slot indices can be resolved to addresses.
     *         Shadows nixlAgent::loadRemoteMD.
     */
    nixl_status_t loadRemoteMD(const nixl_blob_t& remote_metadata, std::string& agent_name);

    // -----------------------------------------------------------------------
    // Staging memory registration
    // -----------------------------------------------------------------------

    /**
     * @brief  Register a staging buffer for the given service mode.
     *
     * Delegates to the marshal backend for transport registration and any
     * library-specific setup (e.g. nvCOMP workspace).  Must be called before
     * metadata exchange.  Re-exchange getLocalMD() / loadRemoteMD() after
     * every register/deregister call.
     *
     * Slot granularity is derived from the backend's queryRequirements(); all
     * algorithms registered for the same mode share the same staging memory.
     *
     * @param  descs        Staging buffer descriptors
     * @param  use          Service mode this memory is staged for
     * @param  extra_params Optional backend hints
     * @return nixl_status_t
     */
    nixl_status_t
    registerServiceMem(const nixl_reg_dlist_t& descs,
                       nixl_service_mode_t     use,
                       const nixl_opt_args_t*  extra_params = nullptr);

    /**
     * @brief  Deregister previously registered staging memory.
     *         Must be followed by metadata re-exchange.
     */
    nixl_status_t
    deregisterServiceMem(const nixl_reg_dlist_t& descs,
                         nixl_service_mode_t     use,
                         const nixl_opt_args_t*  extra_params = nullptr);

    // -----------------------------------------------------------------------
    // Transfer lifecycle — service handle overloads
    // -----------------------------------------------------------------------

    /**
     * @brief  Create a service transfer request from two descriptor lists.
     *
     * In DIRECT mode: delegates to the base nixlAgent createXferReq.
     * In marshal modes: allocates a nixlServiceXferReqH tracking pipeline
     * state for the full RTS→CTS→WRITE→drain→slot-freed exchange.
     *
     * @param  operation    NIXL_WRITE or NIXL_READ
     * @param  local_descs  Local descriptor list
     * @param  remote_descs Remote descriptor list
     * @param  remote_agent Remote agent name
     * @param  req_hndl     [out] Service transfer handle
     * @param  extra_params Optional per-transfer mode/algorithm overrides
     * @return nixl_status_t
     */
    nixl_status_t
    createXferReq(const nixl_xfer_op_t&     operation,
                  const nixl_xfer_dlist_t&  local_descs,
                  const nixl_xfer_dlist_t&  remote_descs,
                  const std::string&        remote_agent,
                  nixlServiceXferReqH*&     req_hndl,
                  const nixl_service_opt_args_t* extra_params = nullptr);

    /**
     * @brief  Create a service transfer request from pre-prepared dlist handles.
     *
     * In DIRECT mode: delegates to the base nixlAgent makeXferReq.
     * In marshal modes: allocates a nixlServiceXferReqH.
     *
     * @param  operation      NIXL_WRITE or NIXL_READ
     * @param  local_side     Pre-prepared local dlist handle
     * @param  local_indices  Descriptor indices to use from local_side
     * @param  remote_side    Pre-prepared remote dlist handle
     * @param  remote_indices Descriptor indices to use from remote_side
     * @param  req_hndl       [out] Service transfer handle
     * @param  extra_params   Optional per-transfer overrides
     * @return nixl_status_t
     */
    nixl_status_t
    makeXferReq(const nixl_xfer_op_t&     operation,
                const nixlDlistH*         local_side,
                const std::vector<int>&   local_indices,
                const nixlDlistH*         remote_side,
                const std::vector<int>&   remote_indices,
                nixlServiceXferReqH*&     req_hndl,
                const nixl_service_opt_args_t* extra_params = nullptr);

    /**
     * @brief  Post a service transfer request, initiating the protocol.
     *
     * In DIRECT mode: delegates to base nixlAgent postXferReq.
     * In marshal modes (WRITE): sends RTS to the receiver.
     * In marshal modes (READ): sends READ_REQ to the remote.
     *
     * @param  req_hndl    Service transfer handle from createXferReq / makeXferReq
     * @param  extra_params Optional per-post overrides
     * @return nixl_status_t  NIXL_IN_PROG or error
     */
    nixl_status_t
    postXferReq(nixlServiceXferReqH*      req_hndl,
                const nixl_service_opt_args_t* extra_params = nullptr);

    /**
     * @brief  Query transfer status, driving service progress in the process.
     *         For marshal modes this is equivalent to a driveService() tick
     *         followed by a status read; callers do not need a separate
     *         driveService() call in a status-polling loop.
     *
     * @param  req_hndl  Service transfer handle
     * @return nixl_status_t  NIXL_SUCCESS (done), NIXL_IN_PROG, or error
     */
    nixl_status_t
    getXferStatus(nixlServiceXferReqH* req_hndl);

    /**
     * @brief  Release a service transfer handle and free all resources.
     *         Releases all owned sub-transfer nixlXferReqH* handles via the
     *         base agent.  Callers must not use req_hndl after this call.
     *
     * @param  req_hndl  Service transfer handle to release
     * @return nixl_status_t
     */
    nixl_status_t
    releaseXferReq(nixlServiceXferReqH* req_hndl);

    // -----------------------------------------------------------------------
    // Service progress
    // -----------------------------------------------------------------------

    /**
     * @brief  Collect user-visible notifications, driving service progress first.
     *
     * The receiver side of a WRITE transfer should call this to collect the
     * final user notification delivered when all data reaches its destination.
     * Initiator side may also call this instead of (or in addition to)
     * polling getXferStatus().
     *
     * Shadows nixlAgent::getNotifs.
     *
     * @param  notifs       [in/out] Appended with non-service notifications
     * @param  extra_params Optional backend filter
     * @return nixl_status_t
     */
    nixl_status_t
    getNotifs(nixl_notifs_t&        notifs,
              const nixl_opt_args_t* extra_params = nullptr);

    // -----------------------------------------------------------------------
    // = delete base-type overloads
    //
    // If a caller holds the handle as nixlXferReqH* (or hasn't yet switched
    // to nixlServiceXferReqH*), these deleted overloads produce a clear
    // compile error:
    //   "error: use of deleted function '... nixlXferReqH* ...'"
    // Fix: declare the handle as nixlServiceXferReqH* (or auto).
    // -----------------------------------------------------------------------

    nixl_status_t createXferReq(const nixl_xfer_op_t&,
                                 const nixl_xfer_dlist_t&,
                                 const nixl_xfer_dlist_t&,
                                 const std::string&,
                                 nixlXferReqH*&,
                                 const nixl_opt_args_t* = nullptr) const = delete;

    nixl_status_t makeXferReq(const nixl_xfer_op_t&,
                               const nixlDlistH*,
                               const std::vector<int>&,
                               const nixlDlistH*,
                               const std::vector<int>&,
                               nixlXferReqH*&,
                               const nixl_opt_args_t* = nullptr) const = delete;

    nixl_status_t postXferReq(nixlXferReqH*,
                               const nixl_opt_args_t* = nullptr) const = delete;

    nixl_status_t getXferStatus(nixlXferReqH*) const = delete;

    nixl_status_t releaseXferReq(nixlXferReqH*) const = delete;

    // These methods require full pipeline knowledge to be meaningful;
    // deleted until service-aware implementations are added in a later phase.
    nixl_status_t estimateXferCost(const nixlXferReqH*,
                                    std::chrono::microseconds&,
                                    std::chrono::microseconds&,
                                    nixl_cost_t&,
                                    const nixl_opt_args_t* = nullptr) const = delete;

    nixl_status_t getXferTelemetry(const nixlXferReqH*,
                                    nixl_xfer_telem_t&) const = delete;

private:
    /**
     * @struct ConstructTag
     * @brief  Carries both the callback-injected config and the pre-created
     *         serviceData shared_ptr through the delegating constructor.
     *
     * Pre-creating serviceData_ before the base nixlAgent constructor runs
     * allows the callback lambda to capture a weak_ptr to it.
     */
    struct ConstructTag {
        nixlServiceAgentConfig              modifiedCfg;
        std::shared_ptr<nixlServiceAgentData> data;
    };

    /**
     * @brief  Pre-create serviceData, inject _NIXLS_* callback, return tag.
     *         Called by the public constructor before delegating.
     */
    static ConstructTag prepare(nixlServiceAgentConfig cfg);

    /** @brief Delegating constructor that receives the pre-built tag. */
    nixlServiceAgent(const std::string& name, ConstructTag&& tag);

    // shared_ptr so the callback lambda (captured as weak_ptr) can outlive
    // a potential early destruction sequence without a dangling raw pointer.
    std::shared_ptr<nixlServiceAgentData> serviceData_;
};

#endif // NIXL_SERVICE_H
