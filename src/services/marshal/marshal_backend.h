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
#ifndef MARSHAL_BACKEND_H
#define MARSHAL_BACKEND_H

#include "nixl_types.h"
#include "nixl_descriptors.h"
#include "nixl_service_types.h"

// Forward declaration — backends receive a reference to the agent for
// sub-transfer registration and posting, but do not own it.
class nixlAgent;

/**
 * @class nixlMarshalBackend
 * @brief Abstract interface for the transformation layer of the marshal sub-service.
 *
 * Each non-DIRECT nixl_service_mode_t maps to one nixlMarshalBackend implementation:
 *
 *   STAGE_BOTH  → staging_backend  (gather/scatter via cudaMemcpyAsync)
 *   COMPRESS    → nvcomp_backend   (Phase 3 — compress/decompress via nvCOMP)
 *
 * The wire protocol (RTS→CTS→NIXL_WRITE→drain→slot-freed) is owned entirely
 * by nixlServiceAgent.  nixlMarshalBackend defines only what varies per mode:
 *   - which memory types are accepted for staging
 *   - how to register/deregister staging buffers
 *   - how to fill a slot from source descriptors (async GPU op)
 *   - how to drain a slot into destination descriptors (async GPU op)
 *   - how to poll for fill/drain completion
 *
 * The slot address is used as a unique key for all checkFill/checkDrain calls;
 * a slot is owned by at most one operation at a time.
 *
 * Resource isolation: each active nixlServiceXferReqH owns its own temporary
 * workspace for the lifetime of the transfer (required by nvCOMP).  Workspaces
 * must never be shared across concurrent transfers.
 */
class nixlMarshalBackend {
public:
    virtual ~nixlMarshalBackend() = default;

    // -----------------------------------------------------------------------
    // Capabilities
    // -----------------------------------------------------------------------

    /**
     * @brief  Memory types that this backend accepts for staging buffers.
     *         Called during registerServiceMem to validate the descriptor list.
     */
    virtual nixl_mem_list_t getSupportedMems() const = 0;

    /**
     * @brief  Query backend slot-size requirements.
     *         Each backend appends its own MarshalSlotReq entries to @p out.
     *         nixlServiceAgent::marshalQuery() merges all backends into one
     *         MarshalRequirements before returning it to the caller.
     */
    virtual nixl_status_t queryRequirements(std::vector<MarshalSlotReq>& out) = 0;

    // -----------------------------------------------------------------------
    // Resource management
    // -----------------------------------------------------------------------

    /**
     * @brief  Register staging memory with the transport and any backend library.
     *         After registration, the buffer is RDMA-accessible via the agent.
     *         Must be called before metadata exchange; re-exchange after every call.
     *
     * @param  descs   Staging memory descriptors (must satisfy getSupportedMems())
     * @param  agent   The base nixlAgent used for transport-level registration
     * @return nixl_status_t
     */
    virtual nixl_status_t
    registerStagingMem(const nixl_reg_dlist_t& descs, nixlAgent& agent) = 0;

    /**
     * @brief  Deregister previously registered staging memory.
     *
     * @param  descs   Same descriptors passed to registerStagingMem
     * @param  agent   The base nixlAgent used for deregistration
     * @return nixl_status_t
     */
    virtual nixl_status_t
    deregisterStagingMem(const nixl_reg_dlist_t& descs, nixlAgent& agent) = 0;

    // -----------------------------------------------------------------------
    // Transformation — async GPU operations
    // -----------------------------------------------------------------------

    /**
     * @brief  Asynchronously fill a staging slot from source descriptors.
     *         Submits the GPU operation and returns immediately.
     *
     * @param  src       Source descriptor list (application buffers)
     * @param  slotAddr  Base address of the target staging slot
     * @param  slotSize  Capacity of the staging slot in bytes
     * @return nixl_status_t  NIXL_IN_PROG on successful submission, or error
     */
    virtual nixl_status_t
    fillSlot(const nixl_xfer_dlist_t& src, void* slotAddr, size_t slotSize) = 0;

    /**
     * @brief  Asynchronously drain a staging slot into destination descriptors.
     *         Submits the GPU operation and returns immediately.
     *
     * @param  slotAddr  Base address of the source staging slot
     * @param  dataSize  Number of valid bytes in the slot (filled or compressed size)
     * @param  dst       Destination descriptor list (application buffers)
     * @return nixl_status_t  NIXL_IN_PROG on successful submission, or error
     */
    virtual nixl_status_t
    drainSlot(void* slotAddr, size_t dataSize, const nixl_xfer_dlist_t& dst) = 0;

    /**
     * @brief  Poll fill completion for the slot at slotAddr.
     *         The slot address is the unique completion key — only one fill
     *         may be in flight per slot at a time.
     *
     * @param  slotAddr  Address of the slot submitted via fillSlot
     * @return nixl_status_t  NIXL_SUCCESS (done), NIXL_IN_PROG (still running),
     *                        or error
     */
    virtual nixl_status_t checkFill(void* slotAddr) = 0;

    /**
     * @brief  Poll drain completion for the slot at slotAddr.
     *         The slot address is the unique completion key — only one drain
     *         may be in flight per slot at a time.
     *
     * @param  slotAddr  Address of the slot submitted via drainSlot
     * @return nixl_status_t  NIXL_SUCCESS (done), NIXL_IN_PROG (still running),
     *                        or error
     */
    virtual nixl_status_t checkDrain(void* slotAddr) = 0;
};

#endif // MARSHAL_BACKEND_H
