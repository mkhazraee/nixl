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

#include "marshal_protocol.h"
#include "marshal_utils.h"
#include "../nixl_service_data.h"
#include "nixl.h"

#include <cstring>

// ===========================================================================
// Wire message builders
// ===========================================================================
// Each builder returns a nixl_blob_t starting with the appropriate prefix.

// WRITE path: Initiator → Receiver
// _NIXLS_RTS_<transferId:8><mode:1><slotSize:8><notifMsg:str><dstDescs>
// mode: nixl_service_mode_t packed as uint8_t; tells receiver which backend to use.
// notifMsg: user-specified notification string; delivered to receiver's user on last chunk.
static nixl_blob_t buildRTS(uint64_t                 transferId,
                              nixl_service_mode_t      mode,
                              uint64_t                 slotSize,
                              const std::string&       notifMsg,
                              const nixl_xfer_dlist_t& dstDescs) {
    nixl_blob_t msg;
    msg.reserve(128);
    msg = NIXLS_RTS_PREFIX;
    packU64(msg, transferId);
    packU8 (msg, static_cast<uint8_t>(mode));
    packU64(msg, slotSize);
    packStr(msg, notifMsg);
    packDlist(msg, dstDescs);
    return msg;
}

// WRITE path: Receiver → Initiator
// _NIXLS_CTS_<transferId:8><slotIndex:4>
// One CTS per pipeline slot.  For Phase-2 (data fits in one slot) exactly one
// CTS is sent.  Slot address is derived by the initiator from remoteStagingPools.
static nixl_blob_t buildCTS(uint64_t transferId, uint32_t slotIndex) {
    nixl_blob_t msg;
    msg.reserve(32);
    msg = NIXLS_CTS_PREFIX;
    packU64(msg, transferId);
    packU32(msg, slotIndex);
    return msg;
}

// Chunk notification attached to NIXL_WRITE sub-transfer.
// _NIXLS_WCHUNK_<transferId:8><chunkN:4><isLast:1>  — fires at receiver (WRITE path)
// _NIXLS_RCHUNK_<transferId:8><chunkN:4><isLast:1>  — fires at initiator (READ path)
// chunkN: per-transfer chunk sequence number (0-based).
// isLast: 1 if this is the final chunk of the transfer; 0 otherwise.
//         No total chunk count is sent in RTS/READ_REQ — the receiver learns the end
//         from this flag, enabling backpressure and double buffering without upfront commitment.
// Phase 2: single chunk per transfer; chunkN is always 0 and isLast is always 1.
static nixl_blob_t buildWriteChunkNotif(uint64_t transferId, uint32_t chunkN, bool isLast) {
    nixl_blob_t msg;
    msg.reserve(32);
    msg = NIXLS_WRITE_CHUNK_PREFIX;
    packU64(msg, transferId);
    packU32(msg, chunkN);
    packU8 (msg, isLast ? 1u : 0u);
    return msg;
}

static nixl_blob_t buildReadChunkNotif(uint64_t transferId, uint32_t chunkN, bool isLast) {
    nixl_blob_t msg;
    msg.reserve(32);
    msg = NIXLS_READ_CHUNK_PREFIX;
    packU64(msg, transferId);
    packU32(msg, chunkN);
    packU8 (msg, isLast ? 1u : 0u);
    return msg;
}

// Slot-freed signal: sent per-chunk after each drain completes, allowing the peer to
// reuse the slot immediately (backpressure and double buffering).
// _NIXLS_WFREE_<transferId:8><slotIndex:4><isLast:1>  — receiver → initiator (WRITE path)
// _NIXLS_RFREE_<transferId:8><slotIndex:4><isLast:1>  — initiator → provider (READ path)
// isLast mirrors the chunk's isLast flag so the peer knows when the transfer is done.
static nixl_blob_t buildWriteSlotFreed(uint64_t transferId, uint32_t slotIndex, bool isLast) {
    nixl_blob_t msg;
    msg.reserve(32);
    msg = NIXLS_WRITE_SLOT_FREED_PREFIX;
    packU64(msg, transferId);
    packU32(msg, slotIndex);
    packU8 (msg, isLast ? 1u : 0u);
    return msg;
}

static nixl_blob_t buildReadSlotFreed(uint64_t transferId, uint32_t slotIndex, bool isLast) {
    nixl_blob_t msg;
    msg.reserve(32);
    msg = NIXLS_READ_SLOT_FREED_PREFIX;
    packU64(msg, transferId);
    packU32(msg, slotIndex);
    packU8 (msg, isLast ? 1u : 0u);
    return msg;
}

// READ path: Initiator → Provider
// _NIXLS_READ_REQ_<transferId:8><slotIndex:4><mode:1><slotSize:8><notifMsg:str><dstDescs>
// slotIndex: initiator's pre-allocated local staging slot the provider must write into.
// mode: nixl_service_mode_t packed as uint8_t; tells provider which backend to use.
// slotSize: initiator's slot size; provider uses it to bound how much it may write.
// notifMsg: user-specified notification string; delivered to provider's user on last RFREE.
static nixl_blob_t buildReadReq(uint64_t                 transferId,
                                 uint32_t                 slotIndex,
                                 nixl_service_mode_t      mode,
                                 uint64_t                 slotSize,
                                 const std::string&       notifMsg,
                                 const nixl_xfer_dlist_t& dstDescs) {
    nixl_blob_t msg;
    msg.reserve(128);
    msg = NIXLS_READ_REQ_PREFIX;
    packU64(msg, transferId);
    packU32(msg, slotIndex);
    packU8 (msg, static_cast<uint8_t>(mode));
    packU64(msg, slotSize);
    packStr(msg, notifMsg);
    packDlist(msg, dstDescs);
    return msg;
}

// ===========================================================================
// Synchronous GPU poll helper
// ===========================================================================

static nixl_status_t syncPoll(nixlMarshalBackend* backend,
                               bool isFill, void* slotAddr) {
    for (;;) {
        nixl_status_t s = isFill ? backend->checkFill(slotAddr)
                                 : backend->checkDrain(slotAddr);
        if (s != NIXL_IN_PROG) return s;
    }
}

// ===========================================================================
// SlotPool helper — find pool and allocate a slot
// ===========================================================================

// Allocate a slot from the first pool that has a free slot.
// Returns the pool pointer and the slot index, or {nullptr, -1} on failure.
static std::pair<SlotPool*, int32_t>
allocFromPools(std::unordered_map<uintptr_t, SlotPool>& pools) {
    for (auto& [_, pool] : pools) {
        int32_t idx = pool.allocSlot();
        if (idx >= 0)
            return {&pool, idx};
    }
    return {nullptr, -1};
}

// ===========================================================================
// Notification callback handlers
// (called from the UCX progress thread; MUST NOT call NIXL agent methods)
// ===========================================================================

// static
void nixlMarshalProtocol::handleRTS(nixlServiceAgentData& d,
                                     const std::string&    remoteAgent,
                                     const nixl_blob_t&    msg) {
    // Parse body (after prefix)
    size_t o = sizeof(NIXLS_RTS_PREFIX) - 1;
    if (o >= msg.size()) return;

    uint64_t              transferId   = unpackU64(msg, o);
    nixl_service_mode_t   mode         = static_cast<nixl_service_mode_t>(unpackU8(msg, o));
    uint64_t              initSlotSize = unpackU64(msg, o);
    std::string           notifMsg     = unpackStr(msg, o);
    nixl_xfer_dlist_t     dstDescs     = unpackDlist(msg, o);

    // Allocate a local staging slot
    std::lock_guard<std::mutex> lk(d.dataMutex);
    auto [pool, slotIdx] = allocFromPools(d.localStagingPools);
    if (!pool) return; // no staging memory registered; silently drop

    // Pipelining: numStages = ceil(initSlotSize / pool->slotSize).
    // Phase 2: single slot only; data fits in one slot (initSlotSize == pool->slotSize).
    (void)initSlotSize; // receiver uses its own pool's slot size; sizes need not match

    // Record inbound transfer state
    InboundTransferState state;
    state.transferId        = transferId;
    state.initiatorAgent    = remoteAgent;
    state.localSlotIndex    = static_cast<uint32_t>(slotIdx);
    state.localSlotBaseAddr = pool->baseAddr;
    state.notifMsg          = std::move(notifMsg);
    state.finalDst.emplace(dstDescs);
    state.mode              = mode;
    d.inboundTransfers[transferId] = std::move(state);

    // Queue CTS — deferred to driveService.
    // Phase 2: single slot, one CTS. Phase 3+: one CTS per allocated slot.
    nixl_blob_t cts = buildCTS(transferId, static_cast<uint32_t>(slotIdx));

    ServiceWorkItem item;
    item.type        = ServiceWorkType::SEND_NOTIF;
    item.targetAgent = remoteAgent;
    item.payload     = std::move(cts);
    item.transferId  = transferId;

    std::lock_guard<std::mutex> wlk(d.workQueueMutex);
    d.workQueue.push_back(std::move(item));
}

// static
void nixlMarshalProtocol::handleCTS(nixlServiceAgentData& d,
                                     const std::string&    remoteAgent,
                                     const nixl_blob_t&    msg) {
    size_t o = sizeof(NIXLS_CTS_PREFIX) - 1;
    if (o >= msg.size()) return;

    uint64_t transferId = unpackU64(msg, o);
    uint32_t slotIndex  = unpackU32(msg, o);

    std::lock_guard<std::mutex> lk(d.dataMutex);

    // Require that metadata has been exchanged before proceeding.
    // The actual remote slot address is resolved in driveQueue POST_WRITE.
    if (!d.remoteStagingPools.count(remoteAgent)) return;

    auto it = d.pendingTransfers.find(transferId);
    if (it == d.pendingTransfers.end()) return;

    nixlServiceXferReqH* handle = it->second;
    auto& extra = *handle->transferExtra_;

    // Build chunk notification string for the NIXL_WRITE sub-transfer (WRITE path, chunk 0).
    // Phase 2: single chunk per transfer, so isLast is always true.
    std::string stageNotif = buildWriteChunkNotif(transferId, /*chunkN=*/0, /*isLast=*/true);

    // Queue FILL_SLOT — GPU fill and subsequent POST_WRITE are handled by driveService.
    ServiceWorkItem item;
    item.type             = ServiceWorkType::FILL_SLOT;
    item.targetAgent      = remoteAgent;
    item.payload          = std::move(stageNotif); // will be attached to NIXL_WRITE
    item.transferId       = transferId;
    item.localSlotBaseAddr= extra.localSlotBaseAddr;
    item.localSlotIndex   = extra.localSlotIndex;
    item.remoteSlotIndex  = slotIndex;
    item.mode             = handle->mode_;
    item.handle           = handle;
    if (extra.localDescs)
        item.fillSrcDescs = extra.localDescs;

    std::lock_guard<std::mutex> wlk(d.workQueueMutex);
    d.workQueue.push_back(std::move(item));
}

// WRITE path: chunk notification fires at the receiver when the NIXL_WRITE sub-transfer lands.
// Queues DRAIN_SLOT; the actual drain runs on driveService (not the callback thread).
// static
void nixlMarshalProtocol::handleWriteChunk(nixlServiceAgentData& d,
                                            const std::string&    remoteAgent,
                                            const nixl_blob_t&    msg) {
    size_t o = sizeof(NIXLS_WRITE_CHUNK_PREFIX) - 1;
    if (o >= msg.size()) return;

    uint64_t transferId = unpackU64(msg, o);
    (void)unpackU32(msg, o); // chunkN — reserved for Phase 4 multi-chunk ordering
    bool     isLast     = unpackU8(msg, o) != 0;

    std::lock_guard<std::mutex> lk(d.dataMutex);

    auto it = d.inboundTransfers.find(transferId);
    if (it == d.inboundTransfers.end()) return;
    InboundTransferState& state = it->second;

    ServiceWorkItem item;
    item.type              = ServiceWorkType::DRAIN_SLOT;
    item.transferId        = transferId;
    item.targetAgent       = state.initiatorAgent; // WFREE → initiator
    item.localSlotBaseAddr = state.localSlotBaseAddr;
    item.localSlotIndex    = state.localSlotIndex;
    item.mode              = state.mode;
    item.isLastChunk       = isLast;
    item.handle            = nullptr; // receiver has no nixlServiceXferReqH

    // Only move notifMsg and finalDst on the last chunk; leave them in place
    // for earlier chunks so subsequent WCHUNK notifications can still find them.
    if (isLast) {
        item.payload       = std::move(state.notifMsg);
        item.drainDstDescs = std::move(state.finalDst);
        d.inboundTransfers.erase(it);
    } else {
        item.drainDstDescs = state.finalDst; // copy for non-last chunks
    }

    std::lock_guard<std::mutex> wlk(d.workQueueMutex);
    d.workQueue.push_back(std::move(item));
}

// READ path: chunk notification fires at the initiator when the provider's NIXL_WRITE lands.
// Queues DRAIN_SLOT; the actual drain runs on driveService (not the callback thread).
// static
void nixlMarshalProtocol::handleReadChunk(nixlServiceAgentData& d,
                                           const std::string&    remoteAgent,
                                           const nixl_blob_t&    msg) {
    size_t o = sizeof(NIXLS_READ_CHUNK_PREFIX) - 1;
    if (o >= msg.size()) return;

    uint64_t transferId = unpackU64(msg, o);
    (void)unpackU32(msg, o); // chunkN — reserved for Phase 4 multi-chunk ordering
    bool     isLast     = unpackU8(msg, o) != 0;

    std::lock_guard<std::mutex> lk(d.dataMutex);

    auto it = d.pendingTransfers.find(transferId);
    if (it == d.pendingTransfers.end()) return;
    nixlServiceXferReqH* handle = it->second;
    auto& extra = *handle->transferExtra_;

    ServiceWorkItem item;
    item.type              = ServiceWorkType::DRAIN_SLOT;
    item.transferId        = transferId;
    item.targetAgent       = remoteAgent; // RFREE → provider
    item.localSlotBaseAddr = extra.localSlotBaseAddr;
    item.localSlotIndex    = extra.localSlotIndex;
    item.mode              = handle->mode_;
    item.isLastChunk       = isLast;
    item.drainDstDescs     = extra.localDescs;
    item.handle            = handle;

    std::lock_guard<std::mutex> wlk(d.workQueueMutex);
    d.workQueue.push_back(std::move(item));
}

// WRITE path: slot-freed arrives at the initiator after the receiver has drained.
// Initiator frees its local staging slot and marks the transfer complete.
// static
void nixlMarshalProtocol::handleWriteSlotFreed(nixlServiceAgentData& d,
                                                const std::string&    /*remoteAgent*/,
                                                const nixl_blob_t&    msg) {
    size_t o = sizeof(NIXLS_WRITE_SLOT_FREED_PREFIX) - 1;
    if (o >= msg.size()) return;

    uint64_t transferId = unpackU64(msg, o);
    (void)unpackU32(msg, o); // slotIndex — for debugging; initiator uses its own state
    bool     isLast     = unpackU8(msg, o) != 0;

    std::lock_guard<std::mutex> lk(d.dataMutex);

    auto it = d.pendingTransfers.find(transferId);
    if (it == d.pendingTransfers.end()) return;
    nixlServiceXferReqH* handle = it->second;
    auto& extra = *handle->transferExtra_;

    // Free the initiator's local staging slot on every WFREE (enables double buffering).
    auto poolIt = d.localStagingPools.find(extra.localSlotBaseAddr);
    if (poolIt != d.localStagingPools.end())
        poolIt->second.freeSlot(extra.localSlotIndex);

    // Mark COMPLETE only after the last chunk's WFREE arrives.
    if (isLast) {
        handle->status_.store(NIXL_SUCCESS, std::memory_order_release);
        handle->stage_.store(nixlServiceXferReqH::Stage::COMPLETE,
                              std::memory_order_release);
    }
}

// READ path: slot-freed arrives at the provider after the initiator has drained.
// Provider frees its local staging slot and queues release of the NIXL_WRITE sub-handle.
// static
void nixlMarshalProtocol::handleReadSlotFreed(nixlServiceAgentData& d,
                                               const std::string&    /*remoteAgent*/,
                                               const nixl_blob_t&    msg) {
    size_t o = sizeof(NIXLS_READ_SLOT_FREED_PREFIX) - 1;
    if (o >= msg.size()) return;

    uint64_t transferId = unpackU64(msg, o);
    (void)unpackU32(msg, o); // slotIndex — for debugging
    bool     isLast     = unpackU8(msg, o) != 0;

    std::lock_guard<std::mutex> lk(d.dataMutex);

    auto it = d.inboundTransfers.find(transferId);
    if (it == d.inboundTransfers.end()) return;
    InboundTransferState& state = it->second;

    // Free the provider's local staging slot on every RFREE (enables double buffering).
    auto poolIt = d.localStagingPools.find(state.localSlotBaseAddr);
    if (poolIt != d.localStagingPools.end())
        poolIt->second.freeSlot(state.localSlotIndex);

    nixlXferReqH* writeSubHndl = state.writeSubHandle;
    std::string   notifMsg     = isLast ? std::move(state.notifMsg) : std::string{};
    if (isLast)
        d.inboundTransfers.erase(it);

    // Queue work items — can't call agent methods from callback thread.
    std::lock_guard<std::mutex> wlk(d.workQueueMutex);

    // Release the NIXL_WRITE sub-handle for this chunk.
    if (writeSubHndl) {
        ServiceWorkItem rel;
        rel.type               = ServiceWorkType::RELEASE_XFER_REQ;
        rel.subHandleToRelease = writeSubHndl;
        d.workQueue.push_back(std::move(rel));
    }

    // Deliver user notification to provider's user only after the last chunk is confirmed
    // received by the initiator (i.e., last RFREE arrived).
    if (isLast && !notifMsg.empty()) {
        ServiceWorkItem notif;
        notif.type        = ServiceWorkType::SEND_NOTIF;
        notif.targetAgent = d.agentName; // self-notification
        notif.payload     = std::move(notifMsg);
        d.workQueue.push_back(std::move(notif));
    }
}

// static
void nixlMarshalProtocol::handleReadReq(nixlServiceAgentData& d,
                                         const std::string&    remoteAgent,
                                         const nixl_blob_t&    msg) {
    size_t o = sizeof(NIXLS_READ_REQ_PREFIX) - 1;
    if (o >= msg.size()) return;

    uint64_t            transferId    = unpackU64(msg, o);
    uint32_t            initSlotIndex = unpackU32(msg, o);
    nixl_service_mode_t mode          = static_cast<nixl_service_mode_t>(unpackU8(msg, o));
    uint64_t            initSlotSize  = unpackU64(msg, o);
    std::string         notifMsg      = unpackStr(msg, o);
    nixl_xfer_dlist_t   dstDescs      = unpackDlist(msg, o);
    (void)initSlotSize; // receiver uses its own pool's slot size; sizes need not match

    std::lock_guard<std::mutex> lk(d.dataMutex);

    // Require that metadata has been exchanged before proceeding.
    // The actual remote slot address is resolved in driveQueue POST_WRITE.
    if (!d.remoteStagingPools.count(remoteAgent)) return;

    auto [pool, slotIdx] = allocFromPools(d.localStagingPools);
    if (!pool) return;

    // Record inbound state (needed for SLOT_FREED cleanup on READ path)
    InboundTransferState state;
    state.transferId        = transferId;
    state.initiatorAgent    = remoteAgent;
    state.localSlotIndex    = static_cast<uint32_t>(slotIdx);
    state.localSlotBaseAddr = pool->baseAddr;
    state.notifMsg          = std::move(notifMsg);
    state.mode              = mode;
    d.inboundTransfers[transferId] = std::move(state);

    // Queue FILL_SLOT — fill provider's staging slot from dstDescs in driveService,
    // then proceed to POST_WRITE (provider's staging → initiator's staging).
    // Phase 2: single chunk per transfer, so isLast is always true.
    nixl_blob_t stageAttachment = buildReadChunkNotif(transferId, /*chunkN=*/0, /*isLast=*/true);

    ServiceWorkItem item;
    item.type             = ServiceWorkType::FILL_SLOT;
    item.targetAgent      = remoteAgent; // initiator
    item.payload          = std::move(stageAttachment);
    item.transferId       = transferId;
    item.localSlotBaseAddr= pool->baseAddr;
    item.localSlotIndex   = static_cast<uint32_t>(slotIdx);
    item.remoteSlotIndex  = initSlotIndex;
    item.mode             = mode;
    item.handle           = nullptr; // provider has no nixlServiceXferReqH
    item.fillSrcDescs     = std::move(dstDescs);

    std::lock_guard<std::mutex> wlk(d.workQueueMutex);
    d.workQueue.push_back(std::move(item));
}

// ===========================================================================
// Top-level callback dispatcher
// ===========================================================================

// static
void nixlMarshalProtocol::dispatch(nixlServiceAgentData&  d,
                                    nixlNotifCallbackArgs&& args) {
    const std::string& msg   = args.notifMessage;
    const std::string& agent = args.remoteAgent;

    if (msg.compare(0, sizeof(NIXLS_RTS_PREFIX) - 1,
                    NIXLS_RTS_PREFIX) == 0) {
        handleRTS(d, agent, msg);
    } else if (msg.compare(0, sizeof(NIXLS_CTS_PREFIX) - 1,
                            NIXLS_CTS_PREFIX) == 0) {
        handleCTS(d, agent, msg);
    } else if (msg.compare(0, sizeof(NIXLS_WRITE_CHUNK_PREFIX) - 1,
                            NIXLS_WRITE_CHUNK_PREFIX) == 0) {
        handleWriteChunk(d, agent, msg);
    } else if (msg.compare(0, sizeof(NIXLS_READ_CHUNK_PREFIX) - 1,
                            NIXLS_READ_CHUNK_PREFIX) == 0) {
        handleReadChunk(d, agent, msg);
    } else if (msg.compare(0, sizeof(NIXLS_WRITE_SLOT_FREED_PREFIX) - 1,
                            NIXLS_WRITE_SLOT_FREED_PREFIX) == 0) {
        handleWriteSlotFreed(d, agent, msg);
    } else if (msg.compare(0, sizeof(NIXLS_READ_SLOT_FREED_PREFIX) - 1,
                            NIXLS_READ_SLOT_FREED_PREFIX) == 0) {
        handleReadSlotFreed(d, agent, msg);
    } else if (msg.compare(0, sizeof(NIXLS_READ_REQ_PREFIX) - 1,
                            NIXLS_READ_REQ_PREFIX) == 0) {
        handleReadReq(d, agent, msg);
    }
}

// ===========================================================================
// post() — send RTS or READ_REQ to initiate a STAGE_BOTH transfer
// ===========================================================================

// static
nixl_status_t nixlMarshalProtocol::post(nixlAgent&                agent,
                                         nixlServiceAgentData&     d,
                                         nixlServiceXferReqH*      handle,
                                         const nixl_service_opt_args_t* extra_params) {
    auto& extra = *handle->transferExtra_;

    // Allocate a local staging slot
    {
        std::lock_guard<std::mutex> lk(d.dataMutex);
        auto [pool, slotIdx] = allocFromPools(d.localStagingPools);
        if (!pool) return NIXL_ERR_NOT_FOUND;
        extra.localSlotIndex    = static_cast<uint32_t>(slotIdx);
        extra.localSlotBaseAddr = pool->baseAddr;
        extra.localSlotSize     = pool->slotSize;
    }

    // Extract user notification string from opt args (carried in wire message to other side)
    if (extra_params && extra_params->notif.has_value())
        extra.notifMsg = extra_params->notif.value();

    if (handle->xferOp_ == NIXL_WRITE) {
        nixl_blob_t rtsMsg = buildRTS(
            handle->transferId_,
            handle->mode_,
            static_cast<uint64_t>(extra.localSlotSize),
            extra.notifMsg,
            extra.remoteDescs ? *extra.remoteDescs
                              : nixl_xfer_dlist_t(DRAM_SEG));

        nixl_status_t ret = agent.genNotif(handle->remoteAgent_, rtsMsg);
        if (ret < 0) return ret;

        handle->stage_.store(nixlServiceXferReqH::Stage::WAITING_CTS,
                              std::memory_order_release);
        handle->status_.store(NIXL_IN_PROG, std::memory_order_release);
        return NIXL_IN_PROG;
    }

    if (handle->xferOp_ == NIXL_READ) {
        nixl_blob_t rrMsg = buildReadReq(
            handle->transferId_,
            extra.localSlotIndex,
            handle->mode_,
            static_cast<uint64_t>(extra.localSlotSize),
            extra.notifMsg,
            extra.remoteDescs ? *extra.remoteDescs
                              : nixl_xfer_dlist_t(DRAM_SEG));

        nixl_status_t ret = agent.genNotif(handle->remoteAgent_, rrMsg);
        if (ret < 0) return ret;

        handle->stage_.store(nixlServiceXferReqH::Stage::WAITING_DATA,
                              std::memory_order_release);
        handle->status_.store(NIXL_IN_PROG, std::memory_order_release);
        return NIXL_IN_PROG;
    }

    return NIXL_ERR_NOT_SUPPORTED;
}

// ===========================================================================
// nixlMarshalProtocol::executePostWrite
// ===========================================================================

// static
void nixlMarshalProtocol::executePostWrite(nixlAgent&            agent,
                                            nixlServiceAgentData& d,
                                            ServiceWorkItem&      item) {
    // Resolve slot addresses from staging pool metadata.
    uintptr_t  localSlotAddr,  remoteSlotAddr;
    size_t     localSlotSize,  remoteSlotSize;
    uint64_t   localSlotDevId, remoteSlotDevId;
    nixl_mem_t localSlotMemType, remoteSlotMemType;
    {
        std::lock_guard<std::mutex> dlk(d.dataMutex);
        auto lit = d.localStagingPools.find(item.localSlotBaseAddr);
        if (lit == d.localStagingPools.end()) return;
        const SlotPool& lp = lit->second;
        localSlotAddr    = lp.slotAddr(item.localSlotIndex);
        localSlotSize    = lp.slotSize;
        localSlotDevId   = lp.devId;
        localSlotMemType = lp.memType;

        auto rit = d.remoteStagingPools.find(item.targetAgent);
        if (rit == d.remoteStagingPools.end()) return;
        const SlotPool& rp = rit->second;
        remoteSlotAddr    = rp.slotAddr(item.remoteSlotIndex);
        remoteSlotSize    = rp.slotSize;
        remoteSlotDevId   = rp.devId;
        remoteSlotMemType = rp.memType;
    }

    // Build staging-slot NIXL_WRITE sub-transfer.
    nixl_xfer_dlist_t local_slot(localSlotMemType);
    local_slot.addDesc(nixlBasicDesc(localSlotAddr, localSlotSize, localSlotDevId));

    nixl_xfer_dlist_t remote_slot(remoteSlotMemType);
    remote_slot.addDesc(nixlBasicDesc(remoteSlotAddr, remoteSlotSize, remoteSlotDevId));

    nixl_opt_args_t opt;
    opt.notif = item.payload;  // stage-complete notification attached to write

    nixlXferReqH* sub_handle = nullptr;
    nixl_status_t ret = agent.createXferReq(
        NIXL_WRITE, local_slot, remote_slot,
        item.targetAgent, sub_handle, &opt);
    if (ret < 0) {
        if (item.handle) {
            item.handle->status_.store(ret, std::memory_order_release);
            item.handle->stage_.store(nixlServiceXferReqH::Stage::ERROR,
                                       std::memory_order_release);
        }
        return;
    }

    ret = agent.postXferReq(sub_handle, &opt);
    if (ret < 0) {
        agent.releaseXferReq(sub_handle);
        if (item.handle) {
            item.handle->status_.store(ret, std::memory_order_release);
            item.handle->stage_.store(nixlServiceXferReqH::Stage::ERROR,
                                       std::memory_order_release);
        }
        return;
    }

    if (item.handle) {
        item.handle->subHandles_.push_back(sub_handle);
        item.handle->stage_.store(nixlServiceXferReqH::Stage::ACTIVE,
                                   std::memory_order_release);
        item.handle->status_.store(NIXL_IN_PROG, std::memory_order_release);
    } else {
        // Provider side (READ path): store sub_handle in inbound state
        // so handleReadSlotFreed can queue its release.
        std::lock_guard<std::mutex> dlk(d.dataMutex);
        auto it = d.inboundTransfers.find(item.transferId);
        if (it != d.inboundTransfers.end())
            it->second.writeSubHandle = sub_handle;
    }
}

// ===========================================================================
// driveQueue() — drain deferred work + poll in-flight sub-handles
// ===========================================================================

// static
void nixlMarshalProtocol::driveQueue(nixlAgent&            agent,
                                      nixlServiceAgentData& d) {
    // Drain the work queue
    std::vector<ServiceWorkItem> batch;
    {
        std::lock_guard<std::mutex> lk(d.workQueueMutex);
        batch.swap(d.workQueue);
    }

    for (auto& item : batch) {
        if (item.type == ServiceWorkType::SEND_NOTIF) {
            agent.genNotif(item.targetAgent, item.payload);
            continue;
        }

        if (item.type == ServiceWorkType::RELEASE_XFER_REQ) {
            if (item.subHandleToRelease)
                agent.releaseXferReq(item.subHandleToRelease);
            continue;
        }

        if (item.type == ServiceWorkType::FILL_SLOT) {
            // Resolve local slot address and perform GPU fill.
            // On success executePostWrite is called directly to post the staging NIXL_WRITE.
            uintptr_t fillSlotAddr;
            size_t    fillSlotSize;
            {
                std::lock_guard<std::mutex> dlk(d.dataMutex);
                auto lit = d.localStagingPools.find(item.localSlotBaseAddr);
                if (lit == d.localStagingPools.end()) continue;
                fillSlotAddr = lit->second.slotAddr(item.localSlotIndex);
                fillSlotSize = lit->second.slotSize;
            }

            auto backendIt = d.marshalBackends.find(item.mode);
            if (backendIt != d.marshalBackends.end() && item.fillSrcDescs) {
                nixlMarshalBackend* backend = backendIt->second.get();
                void* slotPtr = reinterpret_cast<void*>(fillSlotAddr);
                nixl_status_t ret = backend->fillSlot(
                    *item.fillSrcDescs, slotPtr, fillSlotSize);
                if (ret >= 0)
                    ret = syncPoll(backend, /*isFill=*/true, slotPtr);
                if (ret < 0) {
                    if (item.handle) {
                        item.handle->status_.store(ret, std::memory_order_release);
                        item.handle->stage_.store(nixlServiceXferReqH::Stage::ERROR,
                                                   std::memory_order_release);
                    }
                    std::lock_guard<std::mutex> dlk(d.dataMutex);
                    auto lit = d.localStagingPools.find(item.localSlotBaseAddr);
                    if (lit != d.localStagingPools.end())
                        lit->second.freeSlot(item.localSlotIndex);
                    continue;
                }
            }
            // Fill complete — execute the post-write step directly.
            executePostWrite(agent, d, item);
            continue;
        }

        if (item.type == ServiceWorkType::DRAIN_SLOT) {
            // Resolve local slot address
            uintptr_t slotAddr;
            size_t    slotSize;
            {
                std::lock_guard<std::mutex> dlk(d.dataMutex);
                auto lit = d.localStagingPools.find(item.localSlotBaseAddr);
                if (lit == d.localStagingPools.end()) continue;
                slotAddr = lit->second.slotAddr(item.localSlotIndex);
                slotSize = lit->second.slotSize;
            }

            // Perform GPU drain into destination descriptors
            auto backendIt = d.marshalBackends.find(item.mode);
            if (backendIt != d.marshalBackends.end() && item.drainDstDescs) {
                nixlMarshalBackend* backend = backendIt->second.get();
                void* slotPtr = reinterpret_cast<void*>(slotAddr);
                nixl_status_t ret = backend->drainSlot(
                    slotPtr, slotSize, *item.drainDstDescs);
                if (ret >= 0)
                    ret = syncPoll(backend, /*isFill=*/false, slotPtr);
                if (ret < 0) {
                    if (item.handle) {
                        item.handle->status_.store(ret, std::memory_order_release);
                        item.handle->stage_.store(nixlServiceXferReqH::Stage::ERROR,
                                                   std::memory_order_release);
                    }
                    std::lock_guard<std::mutex> dlk(d.dataMutex);
                    auto lit = d.localStagingPools.find(item.localSlotBaseAddr);
                    if (lit != d.localStagingPools.end())
                        lit->second.freeSlot(item.localSlotIndex);
                    continue;
                }
            }

            // Free local staging slot
            {
                std::lock_guard<std::mutex> dlk(d.dataMutex);
                auto lit = d.localStagingPools.find(item.localSlotBaseAddr);
                if (lit != d.localStagingPools.end())
                    lit->second.freeSlot(item.localSlotIndex);
            }

            // Send WFREE (WRITE) or RFREE (READ) per chunk — enables backpressure and
            // double buffering: the peer can reuse the slot as soon as this arrives,
            // without waiting for the full transfer to complete.
            // isLast is mirrored in the freed message so the peer knows when the
            // transfer is done (mark COMPLETE on WRITE initiator; user notif on READ provider).
            const nixl_blob_t freed = (item.handle == nullptr)
                ? buildWriteSlotFreed(item.transferId, item.localSlotIndex, item.isLastChunk)
                : buildReadSlotFreed (item.transferId, item.localSlotIndex, item.isLastChunk);
            agent.genNotif(item.targetAgent, freed);

            // WRITE path: deliver user notification to receiver only after the last chunk
            // has been drained into the final destination.
            if (item.handle == nullptr && item.isLastChunk && !item.payload.empty())
                agent.genNotif(d.agentName, item.payload);

            // READ path: mark COMPLETE after the last chunk's RFREE is sent.
            if (item.handle && item.isLastChunk) {
                item.handle->status_.store(NIXL_SUCCESS, std::memory_order_release);
                item.handle->stage_.store(nixlServiceXferReqH::Stage::COMPLETE,
                                           std::memory_order_release);
            }
            continue;
        }

        if (item.type == ServiceWorkType::POST_WRITE) {
            executePostWrite(agent, d, item);
        }
    }

    // Drive completion of in-flight NIXL_WRITE sub-transfers.
    // When postXferReq returns NIXL_IN_PROG the UCX backend stores the pending
    // stage notification inside the backend handle; getXferStatus() calls checkXfer()
    // which sends the notification once the underlying write finishes.
    {
        std::vector<nixlXferReqH*> toCheck;
        {
            std::lock_guard<std::mutex> dlk(d.dataMutex);
            for (auto& [_, h] : d.pendingTransfers)
                for (nixlXferReqH* sub : h->subHandles_)
                    toCheck.push_back(sub);
            for (auto& [_, state] : d.inboundTransfers)
                if (state.writeSubHandle)
                    toCheck.push_back(state.writeSubHandle);
        }
        for (nixlXferReqH* sub : toCheck)
            agent.getXferStatus(sub);
    }
}
