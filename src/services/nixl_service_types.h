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
#ifndef NIXL_SERVICE_TYPES_H
#define NIXL_SERVICE_TYPES_H

#include "nixl_types.h"
#include "nixl_params.h"
#include "nixl_descriptors.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Forward declarations from nixl core (full definitions not needed in this header)
class nixlXferReqH;
class nixlDlistH;
class nixlServiceAgent;
class nixlMarshalProtocol;

// ---------------------------------------------------------------------------
// Service mode
// ---------------------------------------------------------------------------

/**
 * @enum nixl_service_mode_t
 * @brief Per-transfer operating mode for nixlServiceAgent.
 *        DIRECT passes through unchanged to the base nixlAgent.
 *        Marshal modes route through the staging/marshal sub-service.
 */
enum class nixl_service_mode_t {
    DIRECT,      ///< Passthrough — identical to base nixlAgent behaviour
    STAGE_BOTH,  ///< Marshal: gather/scatter via pre-registered staging slot
    COMPRESS,    ///< Marshal: compress/decompress via staging slot (nvCOMP)
    // STAGE_INIT, STAGE_TARGET, ENC, ENC_COMP — reserved for future phases
};

// ---------------------------------------------------------------------------
// Transfer handle
// ---------------------------------------------------------------------------

/**
 * @class nixlServiceXferReqH
 * @brief Tracks the multi-step pipeline state of a single service transfer.
 *
 * Composition type — does NOT inherit nixlXferReqH.  Internally holds the
 * individual nixlXferReqH* sub-transfer handles for in-flight chunk writes.
 * All resources are owned by the service agent; callers must invoke
 * nixlServiceAgent::releaseXferReq() before the handle goes out of scope.
 */
class nixlServiceXferReqH {
public:
    /**
     * @enum Stage
     * @brief Internal pipeline stage of the transfer.
     */
    enum class Stage {
        IDLE,          ///< Handle created, not yet posted
        WAITING_CTS,   ///< RTS sent; waiting for CTS from receiver (WRITE path)
        WAITING_DATA,  ///< READ_REQ sent; waiting for first chunk (READ path)
        ACTIVE,        ///< In-flight chunk write(s) / drain(s) in progress
        COMPLETE,      ///< All chunks delivered and drained; result is final
        ERROR,         ///< Unrecoverable error; check getStatus() for code
    };

    nixlServiceXferReqH()  = default;
    ~nixlServiceXferReqH() = default;

    // Non-copyable, non-movable — handle identity matters
    nixlServiceXferReqH(const nixlServiceXferReqH&)            = delete;
    nixlServiceXferReqH& operator=(const nixlServiceXferReqH&) = delete;
    nixlServiceXferReqH(nixlServiceXferReqH&&)                 = delete;
    nixlServiceXferReqH& operator=(nixlServiceXferReqH&&)      = delete;

    /** @brief Current pipeline stage. */
    Stage getStage() const noexcept {
        return stage_.load(std::memory_order_acquire);
    }

    /** @brief Last recorded status code. */
    nixl_status_t getStatus() const noexcept {
        return status_.load(std::memory_order_acquire);
    }

private:
    std::atomic<Stage>         stage_{Stage::IDLE};
    std::atomic<nixl_status_t> status_{NIXL_ERR_NOT_POSTED};

    // Sub-transfer handles owned by this service handle; released via agent
    std::vector<nixlXferReqH*> subHandles_;

    // --- Pipeline identity ---
    uint64_t            transferId_{0};
    nixl_service_mode_t mode_{nixl_service_mode_t::DIRECT};
    nixl_xfer_op_t      xferOp_{NIXL_WRITE};
    std::string         remoteAgent_;

    /**
     * @struct TransferExtra
     * @brief  Extended per-transfer state for STAGE_BOTH / COMPRESS modes.
     *         Allocated in createXferReq/makeXferReq, freed in releaseXferReq.
     *
     * For WRITE: localDescs = src (user), remoteDescs = dst (user, goes to receiver via RTS).
     * For READ:  localDescs = dst (user), remoteDescs = src (user, goes to provider via READ_REQ).
     */
    struct TransferExtra {
        // User-visible descriptor lists
        std::optional<nixl_xfer_dlist_t> localDescs;   ///< local src (WRITE) or dst (READ)
        std::optional<nixl_xfer_dlist_t> remoteDescs;  ///< remote dst (WRITE) or src (READ)

        // Local staging slot (allocated in postXferReq from localStagingPools)
        uint32_t   localSlotIndex    {0};
        uintptr_t  localSlotBaseAddr {0};
        size_t     localSlotSize     {0};

        // User-specified notification string from postXferReq opt.notif.
        // Carried in the initial wire message (RTS / READ_REQ) so the other side can
        // deliver it to its own user when the transfer completes on their end.
        std::string notifMsg;
    };
    std::unique_ptr<TransferExtra> transferExtra_;

    friend class nixlServiceAgent;
    friend class nixlMarshalProtocol;
};

// ---------------------------------------------------------------------------
// MarshalRequirements
// ---------------------------------------------------------------------------

/**
 * @struct MarshalSlotReq
 * @brief A single slot-size requirement reported by one marshal backend.
 *
 * mode     — which service mode this entry applies to
 * subKey   — per-mode sub-identifier (e.g. algorithm name); nullopt for modes
 *             with no sub-variants (e.g. STAGE_BOTH)
 * slotSize — recommended staging slot size in bytes; 0 means use the whole
 *             registered buffer as a single slot
 */
struct MarshalSlotReq {
    nixl_service_mode_t        mode;
    std::optional<std::string> subKey;
    size_t                     slotSize{0};
};

/**
 * @struct MarshalRequirements
 * @brief Output of nixlServiceAgent::marshalQuery().
 *
 * Each marshal backend appends its own MarshalSlotReq entries via
 * queryRequirements(); nixlServiceAgent::marshalQuery() merges them all
 * before returning to the caller.  Entries with slotSize=0 carry no sizing
 * constraint (staging); non-zero entries are the backend's recommended chunk size.
 */
struct MarshalRequirements {
    std::vector<MarshalSlotReq> entries;
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/**
 * @struct nixlServiceAgentConfig
 * @brief Extends nixlAgentConfig with service-layer settings.
 */
struct nixlServiceAgentConfig : nixlAgentConfig {
    /** Default mode for transfers that don't specify one via per-transfer opts */
    nixl_service_mode_t defaultMode = nixl_service_mode_t::DIRECT;

    /** Default algorithm sub-key for modes that require one (e.g. "ANS" for COMPRESS); nullopt = backend picks */
    std::optional<std::string> defaultCompAlg;
};

// ---------------------------------------------------------------------------
// Per-transfer options
// ---------------------------------------------------------------------------

/**
 * @struct nixl_service_opt_args_t
 * @brief Extends nixl_opt_args_t with per-transfer service overrides.
 *        Resolution order: opt-args → config default → DIRECT.
 */
struct nixl_service_opt_args_t : nixl_opt_args_t {
    /** Override the service mode for this specific transfer */
    std::optional<nixl_service_mode_t> mode;

    /** Override the algorithm sub-key for this transfer (e.g. "BITCOMP" for COMPRESS) */
    std::optional<std::string> compAlg;
};

// ---------------------------------------------------------------------------
// Wire protocol prefix
// ---------------------------------------------------------------------------
// All _NIXLS_* messages are routed by the registered callback and never
// reach the user's notif_map.  The specific per-message prefixes are
// internal to the marshal layer (see marshal/marshal_utils.h).

constexpr char NIXLS_PREFIX[] = "_NIXLS_";

#endif // NIXL_SERVICE_TYPES_H
