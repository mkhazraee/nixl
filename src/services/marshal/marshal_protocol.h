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
#ifndef NIXL_MARSHAL_PROTOCOL_H
#define NIXL_MARSHAL_PROTOCOL_H

#include "nixl_types.h"

// Forward declarations — full types needed only in marshal_protocol.cpp
class nixlAgent;
class nixlServiceAgentData;
class nixlServiceXferReqH;
struct nixl_service_opt_args_t;
struct nixlNotifCallbackArgs;
struct ServiceWorkItem;

/**
 * @class nixlMarshalProtocol
 * @brief Stateless class implementing the STAGE_BOTH marshal wire protocol.
 *
 * All mutable state lives in nixlServiceAgentData (the pimpl).  This class
 * provides three public entry points used by nixlServiceAgent:
 *
 *   dispatch()    — routes incoming _NIXLS_* notifications to the correct handler
 *   post()        — sends RTS (WRITE) or READ_REQ (READ) to initiate a transfer
 *   driveQueue()  — drains the deferred work queue and polls in-flight sub-handles
 *
 * The private handlers (handleRTS, handleCTS, …) run on the UCX progress
 * thread; they must not call NIXL agent methods — they queue ServiceWorkItems
 * instead.  driveQueue() executes those items on the user's thread.
 */
class nixlMarshalProtocol {
public:
    /** Route an incoming _NIXLS_* notification to the correct handler. */
    static void dispatch(nixlServiceAgentData&  d,
                         nixlNotifCallbackArgs&& args);

    /**
     * @brief Send RTS (NIXL_WRITE) or READ_REQ (NIXL_READ) to kick off a
     *        STAGE_BOTH transfer.  Called from nixlServiceAgent::postXferReq.
     */
    static nixl_status_t post(nixlAgent&                agent,
                               nixlServiceAgentData&     d,
                               nixlServiceXferReqH*      handle,
                               const nixl_service_opt_args_t* extra_params);

    /**
     * @brief Drain the deferred work queue (SEND_NOTIF, FILL_SLOT, DRAIN_SLOT,
     *        POST_WRITE, RELEASE_XFER_REQ) and poll all in-flight NIXL_WRITE
     *        sub-handles.  Called from nixlServiceAgent::driveService.
     */
    static void driveQueue(nixlAgent&            agent,
                           nixlServiceAgentData& d);

private:
    static void handleRTS           (nixlServiceAgentData& d,
                                      const std::string&    remoteAgent,
                                      const nixl_blob_t&    msg);
    static void handleCTS           (nixlServiceAgentData& d,
                                      const std::string&    remoteAgent,
                                      const nixl_blob_t&    msg);
    static void handleWriteChunk    (nixlServiceAgentData& d,
                                      const std::string&    remoteAgent,
                                      const nixl_blob_t&    msg);
    static void handleReadChunk     (nixlServiceAgentData& d,
                                      const std::string&    remoteAgent,
                                      const nixl_blob_t&    msg);
    static void handleWriteSlotFreed(nixlServiceAgentData& d,
                                      const std::string&    remoteAgent,
                                      const nixl_blob_t&    msg);
    static void handleReadSlotFreed (nixlServiceAgentData& d,
                                      const std::string&    remoteAgent,
                                      const nixl_blob_t&    msg);
    static void handleReadReq       (nixlServiceAgentData& d,
                                      const std::string&    remoteAgent,
                                      const nixl_blob_t&    msg);

    /** Build and post the staging-slot NIXL_WRITE sub-transfer.
     *  Called from both FILL_SLOT (after fill completes) and POST_WRITE items. */
    static void executePostWrite    (nixlAgent&            agent,
                                     nixlServiceAgentData& d,
                                     ServiceWorkItem&      item);
};

#endif // NIXL_MARSHAL_PROTOCOL_H
