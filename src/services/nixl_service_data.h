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
#ifndef NIXL_SERVICE_DATA_H
#define NIXL_SERVICE_DATA_H

#include "nixl_service_types.h"
#include "marshal/marshal_backend.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration — full type needed only in nixl_service.cpp
class nixlAgent;

// ---------------------------------------------------------------------------
// SlotPool
// ---------------------------------------------------------------------------

/**
 * @struct SlotPool
 * @brief  Manages a pool of equally-sized staging slots within a contiguous
 *         registered memory region.
 *
 * slotSize and numSlots are derived from the backend's queryRequirements():
 *
 *   slotSize = min(algSlotSizes) if non-empty, else total_buffer_size
 *   numSlots = ceil(total_buffer_size / slotSize)
 *
 * Wire messages carry only slot indices; both sides compute the slot address
 * as baseAddr + slot_idx * slotSize.
 */
struct SlotPool {
    uintptr_t         baseAddr  = 0;
    size_t            slotSize  = 0;
    uint32_t          numSlots  = 0;
    uint64_t          devId     = 0;
    nixl_mem_t        memType   = DRAM_SEG;
    std::vector<bool> inUse;    ///< indexed by slot index; true = in use

    /** @brief Allocate the first free slot. Returns slot index, or -1 if full. */
    int32_t allocSlot() {
        for (uint32_t i = 0; i < numSlots; ++i) {
            if (!inUse[i]) { inUse[i] = true; return static_cast<int32_t>(i); }
        }
        return -1;
    }

    /** @brief Free a previously allocated slot. */
    void freeSlot(uint32_t idx) {
        if (idx < numSlots) inUse[idx] = false;
    }

    /** @brief Address of slot at index idx. */
    uintptr_t slotAddr(uint32_t idx) const {
        return baseAddr + static_cast<uintptr_t>(idx) * slotSize;
    }
};

// ---------------------------------------------------------------------------
// InboundTransferState  (receiver/provider side — no nixlServiceXferReqH)
// ---------------------------------------------------------------------------

/**
 * @struct InboundTransferState
 * @brief  Receiver-side state for an in-flight STAGE_BOTH WRITE or READ_REQ.
 *
 * For a WRITE, the receiver:
 *   1. Allocates a local staging slot on RTS.
 *   2. Sends CTS (queued for driveService).
 *   3. Drains the slot when each WCHUNK notification arrives.
 *   4. Sends WFREE per chunk (queued for driveService); delivers user notif on last chunk.
 *
 * For a READ_REQ, the provider:
 *   1. Allocates a local staging slot on READ_REQ.
 *   2. Fills the slot from srcDescs (queued for driveService).
 *   3. Posts the NIXL_WRITE to the initiator's slot (queued for driveService).
 *   4. Frees the local slot when SLOT_FREED arrives from the initiator.
 */
struct InboundTransferState {
    uint64_t    transferId       {0};
    std::string initiatorAgent;

    // Local staging slot held for this inbound transfer
    uintptr_t  localSlotBaseAddr {0};
    uint32_t   localSlotIndex    {0};

    // Final destination (WRITE path: data drains here on receiver)
    std::optional<nixl_xfer_dlist_t> finalDst;

    // READ path: NIXL_WRITE sub-handle posted by provider; released on SLOT_FREED
    nixlXferReqH* writeSubHandle {nullptr};

    // User-specified notification string delivered to this side's user when transfer completes.
    // WRITE: received in RTS, delivered to receiver's user after drain.
    // READ:  received in READ_REQ, delivered to provider's user after RFREE.
    std::string notifMsg;

    // Marshal mode for this transfer — decoded from wire (RTS or READ_REQ)
    nixl_service_mode_t mode{nixl_service_mode_t::STAGE_BOTH};
};

// ---------------------------------------------------------------------------
// ServiceWorkItem  (deferred NIXL networking operations)
// ---------------------------------------------------------------------------

/**
 * @enum ServiceWorkType
 * @brief Tag for the type of deferred networking work.
 */
enum class ServiceWorkType {
    SEND_NOTIF,       ///< Call genNotif(targetAgent, payload)
    FILL_SLOT,        ///< Call fillSlot+checkFill on a local slot; on completion queue POST_WRITE
    POST_WRITE,       ///< Build and post a NIXL_WRITE sub-transfer (staging slot)
    DRAIN_SLOT,       ///< Call drainSlot+checkDrain; then free slot, send SLOT_FREED, deliver final notif
    RELEASE_XFER_REQ, ///< Call nixlAgent::releaseXferReq(subHandleToRelease)
};

/**
 * @struct ServiceWorkItem
 * @brief  Work deferred from a notification callback to driveService().
 *
 * Callbacks run in the UCX progress thread and must not call NIXL agent
 * functions.  They queue ServiceWorkItems instead; driveService() drains
 * the queue on the user's thread.
 */
struct ServiceWorkItem {
    ServiceWorkType type;
    std::string     targetAgent;
    nixl_blob_t     payload;      ///< SEND_NOTIF: wire message body; FILL_SLOT/POST_WRITE: chunk notif attached to NIXL_WRITE; DRAIN_SLOT (WRITE): user notifMsg delivered on last chunk
    uint64_t        transferId    {0};

    // POST_WRITE / FILL_SLOT fields — slot indices for the staging-slot NIXL_WRITE sub-transfer.
    // driveService() resolves addresses via localStagingPools and remoteStagingPools.
    uintptr_t  localSlotBaseAddr {0};    ///< key into localStagingPools
    uint32_t   localSlotIndex    {0};
    uint32_t   remoteSlotIndex   {0};

    // Marshal mode — used to select the correct backend for FILL_SLOT / DRAIN_SLOT
    nixl_service_mode_t mode{nixl_service_mode_t::STAGE_BOTH};

    // True if this is the last chunk of the transfer; parsed from WCHUNK/RCHUNK wire field.
    // Receiver uses it to decide when to deliver user notification (WRITE) or mark COMPLETE (READ).
    bool isLastChunk{true};

    // Handle to update after POST_WRITE is complete (may be nullptr for provider side)
    nixlServiceXferReqH* handle {nullptr};

    // RELEASE_XFER_REQ: sub-handle to release via nixlAgent::releaseXferReq
    nixlXferReqH* subHandleToRelease {nullptr};

    // FILL_SLOT: source descriptors to gather into the local staging slot.
    // On completion driveService proceeds directly to the POST_WRITE step.
    std::optional<nixl_xfer_dlist_t> fillSrcDescs;

    // DRAIN_SLOT: destination descriptors to scatter from the local staging slot.
    std::optional<nixl_xfer_dlist_t> drainDstDescs;
};

// ---------------------------------------------------------------------------
// nixlServiceAgentData (pimpl for nixlServiceAgent)
// ---------------------------------------------------------------------------

/**
 * @class nixlServiceAgentData
 * @brief Pimpl data bag for nixlServiceAgent.
 *
 * Holds the per-agent service state: staging slot pools (local and remote),
 * the in-flight transfer map, the inbound-transfer map (receiver side),
 * the deferred-work queue, and the marshal backend instance.
 *
 * Wire protocol logic lives in nixlMarshalProtocol; this class only stores
 * state.  The shared_ptr identity is needed so the _NIXLS_* callback lambda
 * can capture a weak_ptr that outlives any early destruction sequence.
 *
 * dataMutex guards all mutable state that may be accessed concurrently
 * from notification callbacks (UCX progress thread) and user threads
 * (driveService / getXferStatus).
 */
class nixlServiceAgentData {
public:
    explicit nixlServiceAgentData(const nixlServiceAgentConfig& cfg);
    ~nixlServiceAgentData() = default;

    // Non-copyable, non-movable
    nixlServiceAgentData(const nixlServiceAgentData&)            = delete;
    nixlServiceAgentData& operator=(const nixlServiceAgentData&) = delete;

    // -----------------------------------------------------------------------
    // Back-pointer to the owning agent (set by nixlServiceAgent constructor).
    // Used by driveService() to call driveQueue and base getNotifs.
    // -----------------------------------------------------------------------

    nixlAgent* agent_ = nullptr;

    // -----------------------------------------------------------------------
    // Progress driver
    // -----------------------------------------------------------------------

    /**
     * @brief  Drain the service work queue and collect user-visible notifications.
     *         Called by nixlServiceAgent::getNotifs() and getXferStatus().
     *         In Phase 4 the background thread body is driveService() in a loop.
     */
    nixl_status_t driveService(nixl_notifs_t&        notifs,
                                const nixl_opt_args_t* extra_params = nullptr);

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    nixlServiceAgentConfig cfg;

    // Agent's own name — needed to deliver final notifications locally (WRITE path)
    std::string agentName;

    // -----------------------------------------------------------------------
    // Staging slot pools
    // -----------------------------------------------------------------------

    /** Our own staging slots registered via registerServiceMem.
     *  Keyed by buffer base address to support multiple registered regions. */
    std::unordered_map<uintptr_t, SlotPool> localStagingPools;

    /** Remote peers' staging slot layout, populated from their service
     *  metadata during loadRemoteMD.  Keyed by remote agent name. */
    std::unordered_map<std::string, SlotPool> remoteStagingPools;

    // -----------------------------------------------------------------------
    // In-flight transfers (initiator side)
    // -----------------------------------------------------------------------

    /** Maps transfer_id → service handle.  Handles are NOT owned here;
     *  owned by the caller until releaseXferReq() is called. */
    std::unordered_map<uint64_t, nixlServiceXferReqH*> pendingTransfers;

    /** Monotonically increasing transfer ID counter. */
    std::atomic<uint64_t> nextTransferId{1};

    // -----------------------------------------------------------------------
    // In-flight transfers (receiver / provider side)
    // -----------------------------------------------------------------------

    /** Maps transfer_id → inbound transfer state for the receiver/provider. */
    std::unordered_map<uint64_t, InboundTransferState> inboundTransfers;

    // -----------------------------------------------------------------------
    // Deferred work queue
    // -----------------------------------------------------------------------

    std::mutex                   workQueueMutex;
    std::vector<ServiceWorkItem> workQueue;

    // -----------------------------------------------------------------------
    // Data mutex — protects pendingTransfers, inboundTransfers, localStagingPools
    // -----------------------------------------------------------------------

    std::mutex dataMutex;

    // -----------------------------------------------------------------------
    // Marshal backend
    // -----------------------------------------------------------------------

    /** Marshal backends keyed by service mode.  Empty for DIRECT-only agents.
     *  Phase 3 adds COMPRESS → nvcomp_backend alongside STAGE_BOTH. */
    std::map<nixl_service_mode_t, std::unique_ptr<nixlMarshalBackend>> marshalBackends;

    friend class nixlServiceAgent;
    friend class nixlMarshalProtocol;
};

#endif // NIXL_SERVICE_DATA_H
