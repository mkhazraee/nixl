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

#include "nixl_service.h"
#include "nixl_service_data.h"
#include "marshal/marshal_utils.h"
#include "marshal/marshal_protocol.h"
#include "marshal/staging/staging_backend.h"

#include <algorithm>

// Service metadata trailer (appended by getLocalMD, stripped by loadRemoteMD):
//   [pool_data: svc_len bytes][svc_len: 4B big-endian][SVC_MD_MAGIC: 4B big-endian]
static constexpr uint32_t SVC_MD_MAGIC = 0x4E584C53u; // "NXLS"

// ===========================================================================
// nixlServiceAgentData
// ===========================================================================

nixlServiceAgentData::nixlServiceAgentData(const nixlServiceAgentConfig& cfg_in)
    : cfg(cfg_in)
{
    // marshalBackends are created lazily in registerServiceMem.
}

nixl_status_t
nixlServiceAgentData::driveService(nixl_notifs_t&        notifs,
                                    const nixl_opt_args_t* extra_params) {
    nixlMarshalProtocol::driveQueue(*agent_, *this);
    // Explicit base-class call avoids re-entering nixlServiceAgent::getNotifs.
    return agent_->nixlAgent::getNotifs(notifs, extra_params);
}

// ===========================================================================
// nixlService::marshalQuery
// ===========================================================================

nixl_status_t nixlService::marshalQuery(MarshalRequirements& reqs) {
    nixlStagingBackend stagingBackend;
    nixl_status_t ret = stagingBackend.queryRequirements(reqs.entries);
    if (ret < 0) return ret;
    // Phase 3: nvcomp_backend nvcompBackend; ret = nvcompBackend.queryRequirements(reqs.entries); ...
    return NIXL_SUCCESS;
}

// ===========================================================================
// nixlServiceAgent — construction
// ===========================================================================

// static
nixlServiceAgent::ConstructTag
nixlServiceAgent::prepare(nixlServiceAgentConfig cfg) {
    // Create serviceData first so the callback lambda can capture a weak_ptr.
    auto data = std::make_shared<nixlServiceAgentData>(cfg);

    cfg.notifCallbacks[NIXLS_PREFIX] =
        [data_weak = std::weak_ptr<nixlServiceAgentData>(data)]
        (nixlNotifCallbackArgs&& args) {
            auto d = data_weak.lock();
            if (!d) return;
            nixlMarshalProtocol::dispatch(*d, std::move(args));
        };

    return ConstructTag{std::move(cfg), std::move(data)};
}

nixlServiceAgent::nixlServiceAgent(const std::string& name, nixlServiceAgentConfig cfg)
    : nixlServiceAgent(name, prepare(std::move(cfg)))
{}

nixlServiceAgent::nixlServiceAgent(const std::string& name, ConstructTag&& tag)
    : nixlAgent(name, tag.modifiedCfg)
    , serviceData_(std::move(tag.data))
{
    serviceData_->agentName = name;
    serviceData_->agent_    = this;
}

nixlServiceAgent::~nixlServiceAgent() = default;

// ===========================================================================
// Staging memory registration
// ===========================================================================

nixl_status_t
nixlServiceAgent::registerServiceMem(const nixl_reg_dlist_t& descs,
                                      nixl_service_mode_t     use,
                                      const nixl_opt_args_t*  extra_params) {
    if (use == nixl_service_mode_t::DIRECT)
        return NIXL_ERR_INVALID_PARAM;

    // Create the marshal backend for this mode on first registration
    if (serviceData_->marshalBackends.find(use) == serviceData_->marshalBackends.end()) {
        if (use == nixl_service_mode_t::STAGE_BOTH)
            serviceData_->marshalBackends[use] = std::make_unique<nixlStagingBackend>();
        else
            return NIXL_ERR_NOT_SUPPORTED;  // COMPRESS backend added in Phase 3
    }

    nixl_status_t ret = serviceData_->marshalBackends[use]->registerStagingMem(descs, *this);
    if (ret < 0) return ret;

    // Derive slot size: min of all non-zero entries, or 0 (whole buffer) if none constrained.
    std::vector<MarshalSlotReq> backendReqs;
    serviceData_->marshalBackends[use]->queryRequirements(backendReqs);
    size_t cfgSlotSz = 0;
    for (auto& req : backendReqs)
        if (req.slotSize > 0 && (cfgSlotSz == 0 || req.slotSize < cfgSlotSz))
            cfgSlotSz = req.slotSize;

    // Build slot pools from the registered descriptors
    std::lock_guard<std::mutex> lk(serviceData_->dataMutex);
    for (int i = 0; i < descs.descCount(); ++i) {
        const nixlBlobDesc& d = descs[i];
        if (d.len == 0) return NIXL_ERR_INVALID_PARAM;

        size_t slotSz = (cfgSlotSz == 0 || cfgSlotSz > d.len) ? d.len : cfgSlotSz;

        SlotPool pool;
        pool.baseAddr = static_cast<uintptr_t>(d.addr);
        pool.devId    = d.devId;
        pool.memType  = descs.getType();
        pool.slotSize = slotSz;
        pool.numSlots = static_cast<uint32_t>((d.len + slotSz - 1) / slotSz); // ceiling
        if (pool.numSlots == 0) return NIXL_ERR_INVALID_PARAM;
        pool.inUse.assign(pool.numSlots, false);

        serviceData_->localStagingPools[pool.baseAddr] = std::move(pool);
    }

    (void)extra_params;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlServiceAgent::deregisterServiceMem(const nixl_reg_dlist_t& descs,
                                        nixl_service_mode_t     use,
                                        const nixl_opt_args_t*  extra_params) {
    if (use == nixl_service_mode_t::DIRECT)
        return NIXL_ERR_INVALID_PARAM;

    auto backendIt = serviceData_->marshalBackends.find(use);
    if (backendIt == serviceData_->marshalBackends.end())
        return NIXL_ERR_NOT_FOUND;

    nixl_status_t ret = backendIt->second->deregisterStagingMem(descs, *this);

    std::lock_guard<std::mutex> lk(serviceData_->dataMutex);
    for (int i = 0; i < descs.descCount(); ++i)
        serviceData_->localStagingPools.erase(
            static_cast<uintptr_t>(descs[i].addr));

    (void)extra_params;
    return ret;
}

// ===========================================================================
// Metadata exchange — service extensions
// ===========================================================================

nixl_status_t nixlServiceAgent::getLocalMD(nixl_blob_t& md) const {
    nixl_status_t ret = nixlAgent::getLocalMD(md);
    if (ret < 0) return ret;

    // Serialize local staging pool layout into a trailer appended to the base blob.
    // Layout: [pool_data: svc_len bytes][svc_len: 4B BE][SVC_MD_MAGIC: 4B BE]
    nixl_blob_t svc;
    {
        std::lock_guard<std::mutex> lk(serviceData_->dataMutex);
        const auto& pools = serviceData_->localStagingPools;

        // Sort by baseAddr for deterministic ordering across both peers
        std::vector<const SlotPool*> ordered;
        ordered.reserve(pools.size());
        for (const auto& [_, p] : pools) ordered.push_back(&p);
        std::sort(ordered.begin(), ordered.end(),
                  [](const SlotPool* a, const SlotPool* b) {
                      return a->baseAddr < b->baseAddr;
                  });

        packU32(svc, static_cast<uint32_t>(ordered.size()));
        for (const SlotPool* p : ordered) {
            packU64(svc, static_cast<uint64_t>(p->baseAddr));
            packU64(svc, static_cast<uint64_t>(p->slotSize));
            packU32(svc, p->numSlots);
            packU64(svc, p->devId);
            packU32(svc, static_cast<uint32_t>(p->memType));
        }
    }

    uint32_t svc_len = static_cast<uint32_t>(svc.size());
    md.append(svc);
    packU32(md, svc_len);
    packU32(md, SVC_MD_MAGIC);
    return NIXL_SUCCESS;
}

nixl_status_t nixlServiceAgent::loadRemoteMD(const nixl_blob_t& remote_metadata,
                                               std::string&       agent_name) {
    // Detect service trailer: last 8 bytes = [svc_len: 4B BE][SVC_MD_MAGIC: 4B BE]
    constexpr size_t TRAILER = 8;
    if (remote_metadata.size() >= TRAILER) {
        size_t o = remote_metadata.size() - TRAILER;
        uint32_t svc_len = unpackU32(remote_metadata, o);  // o → size-4
        uint32_t magic   = unpackU32(remote_metadata, o);  // o → size

        if (magic == SVC_MD_MAGIC && svc_len + TRAILER <= remote_metadata.size()) {
            size_t base_end = remote_metadata.size() - TRAILER - svc_len;
            nixl_blob_t base_md = remote_metadata.substr(0, base_end);

            nixl_status_t ret = nixlAgent::loadRemoteMD(base_md, agent_name);
            if (ret < 0) return ret;

            // Parse service pool data (sorted by baseAddr on the sender side)
            size_t p = base_end;
            uint32_t numPools = unpackU32(remote_metadata, p);

            // Use the first pool as the canonical remote staging pool.
            // Phase 2: single pool per agent; Phase 4+ extends to multi-pool.
            if (numPools > 0) {
                SlotPool rp;
                rp.baseAddr = static_cast<uintptr_t>(unpackU64(remote_metadata, p));
                rp.slotSize = static_cast<size_t>   (unpackU64(remote_metadata, p));
                rp.numSlots = unpackU32(remote_metadata, p);
                rp.devId    = unpackU64(remote_metadata, p);
                rp.memType  = static_cast<nixl_mem_t>(unpackU32(remote_metadata, p));
                // inUse not needed for remote pools

                std::lock_guard<std::mutex> lk(serviceData_->dataMutex);
                serviceData_->remoteStagingPools[agent_name] = std::move(rp);
            }
            return NIXL_SUCCESS;
        }
    }

    // No service trailer — delegate to base class
    return nixlAgent::loadRemoteMD(remote_metadata, agent_name);
}

// ===========================================================================
// Transfer lifecycle
// ===========================================================================

nixl_status_t
nixlServiceAgent::createXferReq(const nixl_xfer_op_t&     operation,
                                 const nixl_xfer_dlist_t&  local_descs,
                                 const nixl_xfer_dlist_t&  remote_descs,
                                 const std::string&        remote_agent,
                                 nixlServiceXferReqH*&     req_hndl,
                                 const nixl_service_opt_args_t* extra_params) {
    nixl_service_mode_t mode = serviceData_->cfg.defaultMode;
    if (extra_params && extra_params->mode.has_value())
        mode = extra_params->mode.value();

    if (mode == nixl_service_mode_t::DIRECT) {
        nixlXferReqH* base_hndl = nullptr;
        nixl_status_t ret = nixlAgent::createXferReq(
            operation, local_descs, remote_descs, remote_agent,
            base_hndl,
            static_cast<const nixl_opt_args_t*>(extra_params));
        if (ret < 0) return ret;

        auto* handle = new nixlServiceXferReqH();
        handle->mode_        = nixl_service_mode_t::DIRECT;
        handle->xferOp_      = operation;
        handle->remoteAgent_ = remote_agent;
        handle->subHandles_.push_back(base_hndl);
        req_hndl = handle;
        return NIXL_SUCCESS;
    }

    if (mode == nixl_service_mode_t::STAGE_BOTH) {
        auto* handle = new nixlServiceXferReqH();
        handle->mode_        = nixl_service_mode_t::STAGE_BOTH;
        handle->xferOp_      = operation;
        handle->remoteAgent_ = remote_agent;
        handle->transferId_  = serviceData_->nextTransferId.fetch_add(1,
                                   std::memory_order_relaxed);

        // Use reset(new ...) rather than make_unique: TransferExtra is private and
        // the new-expression must be in the friend's scope (make_unique's template
        // instantiation is NOT in the friend's scope).
        handle->transferExtra_.reset(new nixlServiceXferReqH::TransferExtra());
        handle->transferExtra_->localDescs.emplace(local_descs);
        handle->transferExtra_->remoteDescs.emplace(remote_descs);

        {
            std::lock_guard<std::mutex> lk(serviceData_->dataMutex);
            serviceData_->pendingTransfers[handle->transferId_] = handle;
        }
        req_hndl = handle;
        return NIXL_SUCCESS;
    }

    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlServiceAgent::makeXferReq(const nixl_xfer_op_t&     operation,
                               const nixlDlistH*         local_side,
                               const std::vector<int>&   local_indices,
                               const nixlDlistH*         remote_side,
                               const std::vector<int>&   remote_indices,
                               nixlServiceXferReqH*&     req_hndl,
                               const nixl_service_opt_args_t* extra_params) {
    nixl_service_mode_t mode = serviceData_->cfg.defaultMode;
    if (extra_params && extra_params->mode.has_value())
        mode = extra_params->mode.value();

    if (mode == nixl_service_mode_t::DIRECT) {
        nixlXferReqH* base_hndl = nullptr;
        nixl_status_t ret = nixlAgent::makeXferReq(
            operation, local_side, local_indices,
            remote_side, remote_indices, base_hndl,
            static_cast<const nixl_opt_args_t*>(extra_params));
        if (ret < 0) return ret;

        auto* handle = new nixlServiceXferReqH();
        handle->mode_   = nixl_service_mode_t::DIRECT;
        handle->xferOp_ = operation;
        handle->subHandles_.push_back(base_hndl);
        req_hndl = handle;
        return NIXL_SUCCESS;
    }

    // STAGE_BOTH / COMPRESS: makeXferReq not yet supported
    // (would need to materialize the dlist handles into descriptor lists)
    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlServiceAgent::postXferReq(nixlServiceXferReqH*      req_hndl,
                               const nixl_service_opt_args_t* extra_params) {
    if (!req_hndl) return NIXL_ERR_INVALID_PARAM;

    if (req_hndl->mode_ == nixl_service_mode_t::DIRECT) {
        if (req_hndl->subHandles_.empty()) return NIXL_ERR_INVALID_PARAM;
        nixl_status_t ret = nixlAgent::postXferReq(
            req_hndl->subHandles_[0],
            static_cast<const nixl_opt_args_t*>(extra_params));
        if (ret >= 0)
            req_hndl->status_.store(ret, std::memory_order_release);
        return ret;
    }

    if (req_hndl->mode_ == nixl_service_mode_t::STAGE_BOTH) {
        if (!req_hndl->transferExtra_) return NIXL_ERR_INVALID_PARAM;
        return nixlMarshalProtocol::post(*this, *serviceData_, req_hndl, extra_params);
    }

    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlServiceAgent::getXferStatus(nixlServiceXferReqH* req_hndl) {
    if (!req_hndl) return NIXL_ERR_INVALID_PARAM;

    if (req_hndl->mode_ == nixl_service_mode_t::DIRECT) {
        if (req_hndl->subHandles_.empty()) return NIXL_ERR_INVALID_PARAM;
        nixl_status_t ret = nixlAgent::getXferStatus(req_hndl->subHandles_[0]);
        req_hndl->status_.store(ret, std::memory_order_release);
        return ret;
    }

    // Drive service progress then read handle status.
    // Notifications surfaced here are discarded; call getNotifs() to collect them.
    nixl_notifs_t unused;
    serviceData_->driveService(unused);
    return req_hndl->status_.load(std::memory_order_acquire);
}

nixl_status_t
nixlServiceAgent::releaseXferReq(nixlServiceXferReqH* req_hndl) {
    if (!req_hndl) return NIXL_ERR_INVALID_PARAM;

    nixl_status_t last_ret = NIXL_SUCCESS;
    for (nixlXferReqH* sub : req_hndl->subHandles_) {
        nixl_status_t ret = nixlAgent::releaseXferReq(sub);
        if (ret < 0) last_ret = ret;
    }
    req_hndl->subHandles_.clear();

    if (req_hndl->mode_ == nixl_service_mode_t::STAGE_BOTH)
        serviceData_->pendingTransfers.erase(req_hndl->transferId_);

    delete req_hndl;
    return last_ret;
}

// ===========================================================================
// getNotifs — drives service progress then collects user-visible notifications
// ===========================================================================

nixl_status_t
nixlServiceAgent::getNotifs(nixl_notifs_t&        notifs,
                             const nixl_opt_args_t* extra_params) {
    return serviceData_->driveService(notifs, extra_params);
}
