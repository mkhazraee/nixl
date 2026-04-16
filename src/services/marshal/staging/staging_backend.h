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
#ifndef NIXL_STAGING_BACKEND_H
#define NIXL_STAGING_BACKEND_H

#include "marshal/marshal_backend.h"

#ifdef NIXL_SERVICE_HAVE_CUDA
#include <cuda_runtime.h>
#include <unordered_map>
#endif
#include <mutex>

/**
 * @class nixlStagingBackend
 * @brief STAGE_BOTH backend: gather/scatter via cudaMemcpyAsync.
 *
 * Maintains one CUDA stream per (slot address, direction); streams are
 * created on first use and destroyed on deregisterStagingMem.
 * Falls back to memcpy when CUDA is not available (DRAM only).
 * See nixlMarshalBackend for the full interface contract.
 */
class nixlStagingBackend : public nixlMarshalBackend {
public:
    nixlStagingBackend();
    ~nixlStagingBackend() override;

    nixl_mem_list_t getSupportedMems() const override;
    nixl_status_t   queryRequirements(std::vector<MarshalSlotReq>& out) override;

    nixl_status_t registerStagingMem  (const nixl_reg_dlist_t& descs, nixlAgent& agent) override;
    nixl_status_t deregisterStagingMem(const nixl_reg_dlist_t& descs, nixlAgent& agent) override;

    nixl_status_t fillSlot (const nixl_xfer_dlist_t& src, void* slotAddr, size_t slotSize) override;
    nixl_status_t drainSlot(void* slotAddr, size_t dataSize, const nixl_xfer_dlist_t& dst) override;
    nixl_status_t checkFill (void* slotAddr) override;
    nixl_status_t checkDrain(void* slotAddr) override;

private:
#ifdef NIXL_SERVICE_HAVE_CUDA
    /** @brief Get-or-create the fill stream for a slot address. */
    cudaStream_t getFillStream(void* slotAddr);
    /** @brief Get-or-create the drain stream for a slot address. */
    cudaStream_t getDrainStream(void* slotAddr);
    /** @brief Destroy all streams whose base address is in [base, base+size). */
    void destroyStreamsForRange(uintptr_t base, size_t size);

    std::mutex                               streamMutex_;
    std::unordered_map<void*, cudaStream_t>  fillStreams_;
    std::unordered_map<void*, cudaStream_t>  drainStreams_;
#endif
};

#endif // NIXL_STAGING_BACKEND_H
