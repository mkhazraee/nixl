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
 * @file nixl_service_example.cpp
 * @brief Single-process integration test for nixlServiceAgent STAGE_BOTH mode.
 *
 * Two agents (A1, A2) run in the same process and perform a WRITE then READ
 * using STAGE_BOTH staging:
 *
 *   WRITE: A1 gathers two non-contiguous 64-byte src chunks into its staging
 *          slot, sends a NIXL_WRITE to A2's staging slot, A2 scatters to its
 *          non-contiguous dst chunks.
 *
 *   READ:  A1 tells A2 which src chunks to gather, A2 fills its staging slot
 *          from those chunks, sends a NIXL_WRITE to A1's staging slot, A1
 *          scatters into its dst chunks.
 *
 * Requires: CUDA (staging_backend uses cudaMemcpyAsync) + UCX transport.
 */

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "nixl_service.h"
#include "test_utils.h"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static const std::string kAgent1Name = "SvcAgent1";
static const std::string kAgent2Name = "SvcAgent2";

// Two 64-byte chunks; source has a 64-byte gap between them
static const size_t kChunkBytes  = 64;
static const size_t kDataBufSize = 256;   // src/dst buffer allocation
static const size_t kStagingSize = 512;   // staging slot; must be >= 2*kChunkBytes

// Fill patterns
static const uint8_t kPatternA = 0xAA;
static const uint8_t kPatternB = 0xBB;
static const uint8_t kPatternC = 0xCC;
static const uint8_t kPatternD = 0xDD;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void drain_protocol(nixlServiceAgent& A1, nixlServiceAgent& A2, int iters = 20) {
    nixl_notifs_t n1, n2;
    for (int i = 0; i < iters; ++i) {
        A1.getNotifs(n1);
        A2.getNotifs(n2);
    }
}

static void poll_until_done(nixlServiceAgent& A1, nixlServiceAgent& A2,
                             nixlServiceXferReqH* req) {
    nixl_notifs_t n2;
    nixl_status_t status;
    do {
        A2.getNotifs(n2);
        status = A1.getXferStatus(req);
        nixl_exit_on_failure(status >= NIXL_SUCCESS, "Transfer error", kAgent1Name);
    } while (status != NIXL_SUCCESS);
}

// ---------------------------------------------------------------------------
// test_write: A1 WRITE (non-contiguous src) → A2 (non-contiguous dst)
// ---------------------------------------------------------------------------

static void test_write(nixlServiceAgent& A1, nixlServiceAgent& A2) {
    std::cout << "\n=== test_write ===\n";

    // Source buffer: chunk0 at +0 (0xAA), chunk1 at +128 (0xBB), gap at +64
    void* src = calloc(1, kDataBufSize);
    memset(static_cast<char*>(src) + 0,   kPatternA, kChunkBytes);
    memset(static_cast<char*>(src) + 128, kPatternB, kChunkBytes);

    // Destination buffer: two contiguous chunks at +0 and +64 (zero-filled)
    void* dst = calloc(1, kDataBufSize);

    // Non-contiguous source descriptors (64-byte gap between chunks)
    nixl_xfer_dlist_t src_descs(DRAM_SEG);
    src_descs.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(src) + 0,   kChunkBytes, 0));
    src_descs.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(src) + 128, kChunkBytes, 0));

    // Non-contiguous destination descriptors on A2's side
    // (offset by different amounts to exercise scatter)
    nixl_xfer_dlist_t dst_descs(DRAM_SEG);
    dst_descs.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(dst) + 0,  kChunkBytes, 0));
    dst_descs.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(dst) + 64, kChunkBytes, 0));

    // Create and post WRITE: A1 is initiator, A2 is receiver
    nixlServiceXferReqH* req = nullptr;
    nixl_status_t ret = A1.createXferReq(NIXL_WRITE, src_descs, dst_descs,
                                          kAgent2Name, req);
    nixl_exit_on_failure(ret, "createXferReq WRITE failed", kAgent1Name);

    ret = A1.postXferReq(req);
    nixl_exit_on_failure(ret >= NIXL_SUCCESS, "postXferReq WRITE failed", kAgent1Name);

    poll_until_done(A1, A2, req);

    // Let SLOT_FREED fully propagate so A2's staging slot is freed
    drain_protocol(A1, A2);

    // Verify: dst[0..63] = 0xAA, dst[64..127] = 0xBB
    const auto* d = static_cast<const uint8_t*>(dst);
    for (size_t i = 0; i < kChunkBytes; ++i) {
        nixl_exit_on_failure(d[i]             == kPatternA, "WRITE chunk0 mismatch", kAgent1Name);
        nixl_exit_on_failure(d[kChunkBytes+i] == kPatternB, "WRITE chunk1 mismatch", kAgent1Name);
    }

    A1.releaseXferReq(req);

    free(src);
    free(dst);

    std::cout << "test_write PASSED\n";
}

// ---------------------------------------------------------------------------
// test_read: A1 READ from A2's non-contiguous src → A1's non-contiguous dst
// ---------------------------------------------------------------------------

static void test_read(nixlServiceAgent& A1, nixlServiceAgent& A2) {
    std::cout << "\n=== test_read ===\n";

    // A2 is the data provider: two chunks with a gap
    void* a2_src = calloc(1, kDataBufSize);
    memset(static_cast<char*>(a2_src) + 0,   kPatternC, kChunkBytes);
    memset(static_cast<char*>(a2_src) + 128, kPatternD, kChunkBytes);

    // A1 is the initiator: gathers into two contiguous destination chunks
    void* a1_dst = calloc(1, kDataBufSize);

    // local_descs  = where A1 wants the data to land (scatter target)
    nixl_xfer_dlist_t local_descs(DRAM_SEG);
    local_descs.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(a1_dst) + 0,  kChunkBytes, 0));
    local_descs.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(a1_dst) + 64, kChunkBytes, 0));

    // remote_descs = what A1 wants to read from A2 (A2's src addresses)
    nixl_xfer_dlist_t remote_descs(DRAM_SEG);
    remote_descs.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(a2_src) + 0,   kChunkBytes, 0));
    remote_descs.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(a2_src) + 128, kChunkBytes, 0));

    // Create and post READ: A1 reads from A2
    nixlServiceXferReqH* req = nullptr;
    nixl_status_t ret = A1.createXferReq(NIXL_READ, local_descs, remote_descs,
                                          kAgent2Name, req);
    nixl_exit_on_failure(ret, "createXferReq READ failed", kAgent1Name);

    ret = A1.postXferReq(req);
    nixl_exit_on_failure(ret >= NIXL_SUCCESS, "postXferReq READ failed", kAgent1Name);

    poll_until_done(A1, A2, req);

    // Let SLOT_FREED and provider-side handle release propagate
    drain_protocol(A1, A2);

    // Verify: a1_dst[0..63] = 0xCC, a1_dst[64..127] = 0xDD
    const auto* d = static_cast<const uint8_t*>(a1_dst);
    for (size_t i = 0; i < kChunkBytes; ++i) {
        nixl_exit_on_failure(d[i]             == kPatternC, "READ chunk0 mismatch", kAgent1Name);
        nixl_exit_on_failure(d[kChunkBytes+i] == kPatternD, "READ chunk1 mismatch", kAgent1Name);
    }

    A1.releaseXferReq(req);

    free(a2_src);
    free(a1_dst);

    std::cout << "test_read PASSED\n";
}

// ---------------------------------------------------------------------------
// test_direct: A3/A4 with DIRECT mode — verifies pure passthrough behaviour
// ---------------------------------------------------------------------------

static void test_direct(const std::string& backend) {
    std::cout << "\n=== test_direct (DIRECT passthrough) ===\n";

    static const std::string kA3 = "DirectAgent1";
    static const std::string kA4 = "DirectAgent2";

    // DIRECT mode: no staging memory or protocol — pure nixlAgent delegation
    nixlServiceAgentConfig dcfg;
    dcfg.useProgThread = true;
    dcfg.defaultMode   = nixl_service_mode_t::DIRECT;

    nixlServiceAgent A3(kA3, dcfg);
    nixlServiceAgent A4(kA4, dcfg);

    nixl_b_params_t p3, p4;
    nixl_mem_list_t m3, m4;
    nixlBackendH *b3 = nullptr, *b4 = nullptr;
    nixl_exit_on_failure(A3.getPluginParams(backend, m3, p3), "getPluginParams A3", kA3);
    nixl_exit_on_failure(A4.getPluginParams(backend, m4, p4), "getPluginParams A4", kA4);
    nixl_exit_on_failure(A3.createBackend(backend, p3, b3), "createBackend A3", kA3);
    nixl_exit_on_failure(A4.createBackend(backend, p4, b4), "createBackend A4", kA4);

    static const size_t kLen = 128;
    void* src = malloc(kLen);
    void* dst = calloc(1, kLen);
    nixl_exit_on_failure(src != nullptr && dst != nullptr, "malloc failed", kA3);
    memset(src, 0xAB, kLen);

    nixl_opt_args_t reg3, reg4;
    reg3.backends.push_back(b3);
    reg4.backends.push_back(b4);

    nixl_reg_dlist_t dl3(DRAM_SEG), dl4(DRAM_SEG);
    dl3.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(src), kLen, 0));
    dl4.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(dst), kLen, 0));

    nixl_exit_on_failure(A3.registerMem(dl3, &reg3), "registerMem A3", kA3);
    nixl_exit_on_failure(A4.registerMem(dl4, &reg4), "registerMem A4", kA4);

    std::string meta3, meta4, rname;
    nixl_exit_on_failure(A3.getLocalMD(meta3), "getLocalMD A3", kA3);
    nixl_exit_on_failure(A4.getLocalMD(meta4), "getLocalMD A4", kA4);
    nixl_exit_on_failure(A3.loadRemoteMD(meta4, rname), "loadRemoteMD A3", kA3);
    nixl_exit_on_failure(A4.loadRemoteMD(meta3, rname), "loadRemoteMD A4", kA4);

    nixl_xfer_dlist_t xsrc(DRAM_SEG), xdst(DRAM_SEG);
    xsrc.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(src), kLen, 0));
    xdst.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(dst), kLen, 0));

    // Pass notification through the DIRECT path
    nixl_service_opt_args_t xopts;
    xopts.notif = "direct_done";

    nixlServiceXferReqH* req = nullptr;
    nixl_exit_on_failure(A3.createXferReq(NIXL_WRITE, xsrc, xdst, kA4, req, &xopts),
                         "createXferReq DIRECT", kA3);

    nixl_status_t status = A3.postXferReq(req);
    nixl_exit_on_failure(status >= NIXL_SUCCESS, "postXferReq DIRECT", kA3);

    nixl_notifs_t notifs;
    int n_notifs = 0;
    nixl_status_t nret = NIXL_SUCCESS;
    do {
        if (status != NIXL_SUCCESS) status = A3.getXferStatus(req);
        nixl_exit_on_failure(status >= NIXL_SUCCESS, "getXferStatus DIRECT", kA3);
        if (n_notifs == 0) {
            nret = A4.getNotifs(notifs);
            nixl_exit_on_failure(nret, "getNotifs A4", kA4);
            n_notifs = static_cast<int>(notifs.size());
        }
    } while (status != NIXL_SUCCESS || n_notifs == 0);

    auto& a3_notifs = notifs[kA3];
    nixl_exit_on_failure(a3_notifs.size() == 1,             "wrong notif count",  kA4);
    nixl_exit_on_failure(a3_notifs.front() == "direct_done","wrong notif string", kA4);

    const auto* s = static_cast<const uint8_t*>(src);
    const auto* d = static_cast<const uint8_t*>(dst);
    for (size_t i = 0; i < kLen; ++i)
        nixl_exit_on_failure(d[i] == s[i], "DIRECT data mismatch", kA3);

    A3.releaseXferReq(req);
    nixl_exit_on_failure(A3.deregisterMem(dl3, &reg3), "deregisterMem A3", kA3);
    nixl_exit_on_failure(A4.deregisterMem(dl4, &reg4), "deregisterMem A4", kA4);
    nixl_exit_on_failure(A3.invalidateRemoteMD(kA4), "invalidateRemoteMD A3", kA3);
    nixl_exit_on_failure(A4.invalidateRemoteMD(kA3), "invalidateRemoteMD A4", kA4);
    free(src);
    free(dst);

    std::cout << "test_direct PASSED\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    std::string backend = "UCX";
    if (argc > 1) backend = argv[1];

    // ---- Construct agents ----
    nixlServiceAgentConfig cfg;
    cfg.useProgThread = true;
    cfg.defaultMode   = nixl_service_mode_t::STAGE_BOTH;

    nixlServiceAgent A1(kAgent1Name, cfg);
    nixlServiceAgent A2(kAgent2Name, cfg);

    // ---- Initialize transport backend ----
    nixl_b_params_t init1, init2;
    nixl_mem_list_t mems1, mems2;
    nixlBackendH *bknd1, *bknd2;

    nixl_exit_on_failure(A1.getPluginParams(backend, mems1, init1),
                         "getPluginParams A1", kAgent1Name);
    nixl_exit_on_failure(A2.getPluginParams(backend, mems2, init2),
                         "getPluginParams A2", kAgent2Name);
    nixl_exit_on_failure(A1.createBackend(backend, init1, bknd1),
                         "createBackend A1", kAgent1Name);
    nixl_exit_on_failure(A2.createBackend(backend, init2, bknd2),
                         "createBackend A2", kAgent2Name);

    // ---- Register staging buffers ----
    // Staging buffers are registered with the transport so NIXL_WRITE can
    // target them.  CUDA will gather/scatter application data into/out of
    // these slots via cudaMemcpyAsync (cudaMemcpyDefault handles DRAM↔DRAM).
    void* staging1 = malloc(kStagingSize);
    void* staging2 = malloc(kStagingSize);
    nixl_exit_on_failure(staging1 != nullptr, "staging malloc A1", kAgent1Name);
    nixl_exit_on_failure(staging2 != nullptr, "staging malloc A2", kAgent2Name);

    nixl_reg_dlist_t stage_dlist1(DRAM_SEG), stage_dlist2(DRAM_SEG);
    stage_dlist1.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(staging1), kStagingSize, 0));
    stage_dlist2.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(staging2), kStagingSize, 0));

    nixl_exit_on_failure(
        A1.registerServiceMem(stage_dlist1, nixl_service_mode_t::STAGE_BOTH),
        "registerServiceMem A1", kAgent1Name);
    nixl_exit_on_failure(
        A2.registerServiceMem(stage_dlist2, nixl_service_mode_t::STAGE_BOTH),
        "registerServiceMem A2", kAgent2Name);

    std::cout << "Staging buffers registered\n";

    // ---- Exchange metadata ----
    // Both agents load each other's metadata so that:
    //   A1 can NIXL_WRITE into A2's staging slot (A1 needs A2's memory info).
    //   A2 can send genNotif(CTS/SLOT_FREED) back to A1 (A2 needs A1's connection).
    //   A2 can NIXL_WRITE into A1's staging slot on READ path (A2 needs A1's memory info).
    std::string meta1, meta2, remote_name;
    nixl_exit_on_failure(A1.getLocalMD(meta1), "getLocalMD A1", kAgent1Name);
    nixl_exit_on_failure(A2.getLocalMD(meta2), "getLocalMD A2", kAgent2Name);
    nixl_exit_on_failure(A1.loadRemoteMD(meta2, remote_name), "loadRemoteMD A1", kAgent1Name);
    nixl_exit_on_failure(A2.loadRemoteMD(meta1, remote_name), "loadRemoteMD A2", kAgent2Name);

    std::cout << "Metadata exchanged\n";

    // ---- Run tests ----
    test_write(A1, A2);
    test_read (A1, A2);
    test_direct(backend);

    // ---- Cleanup ----
    nixl_exit_on_failure(
        A1.deregisterServiceMem(stage_dlist1, nixl_service_mode_t::STAGE_BOTH),
        "deregisterServiceMem A1", kAgent1Name);
    nixl_exit_on_failure(
        A2.deregisterServiceMem(stage_dlist2, nixl_service_mode_t::STAGE_BOTH),
        "deregisterServiceMem A2", kAgent2Name);

    nixl_exit_on_failure(A1.invalidateRemoteMD(kAgent2Name), "invalidateRemoteMD A1", kAgent1Name);
    nixl_exit_on_failure(A2.invalidateRemoteMD(kAgent1Name), "invalidateRemoteMD A2", kAgent2Name);

    free(staging1);
    free(staging2);

    std::cout << "\nAll tests PASSED\n";
    return 0;
}
