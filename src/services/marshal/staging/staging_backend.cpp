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

#include "staging_backend.h"
#include "nixl.h"

#ifndef NIXL_SERVICE_HAVE_CUDA
#include <cstring>  // memcpy
#endif

// ---------------------------------------------------------------------------
// nixlStagingBackend
// ---------------------------------------------------------------------------

nixlStagingBackend::nixlStagingBackend()  = default;
nixlStagingBackend::~nixlStagingBackend() {
#ifdef NIXL_SERVICE_HAVE_CUDA
    std::lock_guard<std::mutex> lk(streamMutex_);
    for (auto& [_, s] : fillStreams_)  cudaStreamDestroy(s);
    for (auto& [_, s] : drainStreams_) cudaStreamDestroy(s);
#endif
}

nixl_status_t nixlStagingBackend::queryRequirements(std::vector<MarshalSlotReq>& out) {
    // Staging has no sub-variant; slotSize=0 means use the whole registered buffer.
    out.push_back({nixl_service_mode_t::STAGE_BOTH, std::nullopt, 0});
    return NIXL_SUCCESS;
}

nixl_mem_list_t nixlStagingBackend::getSupportedMems() const {
#ifdef NIXL_SERVICE_HAVE_CUDA
    return {DRAM_SEG, VRAM_SEG};
#else
    return {DRAM_SEG};
#endif
}

nixl_status_t
nixlStagingBackend::registerStagingMem(const nixl_reg_dlist_t& descs, nixlAgent& agent) {
    return agent.registerMem(descs);
}

nixl_status_t
nixlStagingBackend::deregisterStagingMem(const nixl_reg_dlist_t& descs, nixlAgent& agent) {
#ifdef NIXL_SERVICE_HAVE_CUDA
    for (int i = 0; i < descs.descCount(); ++i)
        destroyStreamsForRange(descs[i].addr, descs[i].len);
#endif
    return agent.deregisterMem(descs);
}

// ---------------------------------------------------------------------------
// fillSlot — gather src descriptors into contiguous slotAddr
// ---------------------------------------------------------------------------

nixl_status_t
nixlStagingBackend::fillSlot(const nixl_xfer_dlist_t& src,
                              void*                     slotAddr,
                              size_t                    slotSize) {
    size_t offset = 0;
    for (int i = 0; i < src.descCount(); ++i) {
        const auto& d = src[i];
        if (offset + d.len > slotSize)
            return NIXL_ERR_INVALID_PARAM;

#ifdef NIXL_SERVICE_HAVE_CUDA
        cudaStream_t stream = getFillStream(slotAddr);
        cudaError_t err = cudaMemcpyAsync(
            static_cast<char*>(slotAddr) + offset,
            reinterpret_cast<const void*>(d.addr),
            d.len,
            cudaMemcpyDefault,
            stream);
        if (err != cudaSuccess)
            return NIXL_ERR_BACKEND;
#else
        memcpy(static_cast<char*>(slotAddr) + offset,
               reinterpret_cast<const void*>(d.addr),
               d.len);
#endif
        offset += d.len;
    }
#ifdef NIXL_SERVICE_HAVE_CUDA
    return NIXL_IN_PROG;  // async — caller must checkFill
#else
    return NIXL_SUCCESS;  // synchronous memcpy — already complete
#endif
}

// ---------------------------------------------------------------------------
// drainSlot — scatter slotAddr into dst descriptors
// ---------------------------------------------------------------------------

nixl_status_t
nixlStagingBackend::drainSlot(void*                     slotAddr,
                               size_t                    dataSize,
                               const nixl_xfer_dlist_t& dst) {
    size_t offset = 0;
    for (int i = 0; i < dst.descCount(); ++i) {
        const auto& d = dst[i];
        if (offset + d.len > dataSize)
            return NIXL_ERR_INVALID_PARAM;

#ifdef NIXL_SERVICE_HAVE_CUDA
        cudaStream_t stream = getDrainStream(slotAddr);
        cudaError_t err = cudaMemcpyAsync(
            reinterpret_cast<void*>(d.addr),
            static_cast<const char*>(slotAddr) + offset,
            d.len,
            cudaMemcpyDefault,
            stream);
        if (err != cudaSuccess)
            return NIXL_ERR_BACKEND;
#else
        memcpy(reinterpret_cast<void*>(d.addr),
               static_cast<const char*>(slotAddr) + offset,
               d.len);
#endif
        offset += d.len;
    }
#ifdef NIXL_SERVICE_HAVE_CUDA
    return NIXL_IN_PROG;
#else
    return NIXL_SUCCESS;
#endif
}

// ---------------------------------------------------------------------------
// checkFill / checkDrain
// ---------------------------------------------------------------------------

nixl_status_t nixlStagingBackend::checkFill(void* slotAddr) {
#ifdef NIXL_SERVICE_HAVE_CUDA
    cudaStream_t stream;
    {
        std::lock_guard<std::mutex> lk(streamMutex_);
        auto it = fillStreams_.find(slotAddr);
        if (it == fillStreams_.end()) return NIXL_SUCCESS;
        stream = it->second;
    }
    cudaError_t err = cudaStreamQuery(stream);
    if (err == cudaSuccess)       return NIXL_SUCCESS;
    if (err == cudaErrorNotReady) return NIXL_IN_PROG;
    return NIXL_ERR_BACKEND;
#else
    (void)slotAddr;
    return NIXL_SUCCESS;  // memcpy is synchronous — always complete
#endif
}

nixl_status_t nixlStagingBackend::checkDrain(void* slotAddr) {
#ifdef NIXL_SERVICE_HAVE_CUDA
    cudaStream_t stream;
    {
        std::lock_guard<std::mutex> lk(streamMutex_);
        auto it = drainStreams_.find(slotAddr);
        if (it == drainStreams_.end()) return NIXL_SUCCESS;
        stream = it->second;
    }
    cudaError_t err = cudaStreamQuery(stream);
    if (err == cudaSuccess)       return NIXL_SUCCESS;
    if (err == cudaErrorNotReady) return NIXL_IN_PROG;
    return NIXL_ERR_BACKEND;
#else
    (void)slotAddr;
    return NIXL_SUCCESS;
#endif
}

// ---------------------------------------------------------------------------
// Private helpers (CUDA only)
// ---------------------------------------------------------------------------

#ifdef NIXL_SERVICE_HAVE_CUDA
cudaStream_t nixlStagingBackend::getFillStream(void* slotAddr) {
    std::lock_guard<std::mutex> lk(streamMutex_);
    auto it = fillStreams_.find(slotAddr);
    if (it != fillStreams_.end()) return it->second;
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    fillStreams_[slotAddr] = s;
    return s;
}

cudaStream_t nixlStagingBackend::getDrainStream(void* slotAddr) {
    std::lock_guard<std::mutex> lk(streamMutex_);
    auto it = drainStreams_.find(slotAddr);
    if (it != drainStreams_.end()) return it->second;
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    drainStreams_[slotAddr] = s;
    return s;
}

void nixlStagingBackend::destroyStreamsForRange(uintptr_t base, size_t size) {
    std::lock_guard<std::mutex> lk(streamMutex_);
    for (auto it = fillStreams_.begin(); it != fillStreams_.end(); ) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(it->first);
        if (addr >= base && addr < base + size) {
            cudaStreamDestroy(it->second);
            it = fillStreams_.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = drainStreams_.begin(); it != drainStreams_.end(); ) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(it->first);
        if (addr >= base && addr < base + size) {
            cudaStreamDestroy(it->second);
            it = drainStreams_.erase(it);
        } else {
            ++it;
        }
    }
}
#endif // NIXL_SERVICE_HAVE_CUDA
