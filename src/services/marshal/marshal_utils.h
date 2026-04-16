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
#ifndef NIXL_MARSHAL_UTILS_H
#define NIXL_MARSHAL_UTILS_H

#include "nixl_types.h"
#include "nixl_descriptors.h"

#include <cstdint>
#include <string>

// ---------------------------------------------------------------------------
// Wire protocol message prefixes
// ---------------------------------------------------------------------------
// All _NIXLS_* messages are routed by the _NIXLS_ callback and never reach
// the user's notif_map.  These constants are internal to the marshal layer;
// only NIXLS_PREFIX (in nixl_service_types.h) is visible outside.

constexpr char NIXLS_RTS_PREFIX[]              = "_NIXLS_RTS_";
constexpr char NIXLS_CTS_PREFIX[]              = "_NIXLS_CTS_";
constexpr char NIXLS_READ_REQ_PREFIX[]         = "_NIXLS_READ_REQ_";
// Chunk notification attached to NIXL_WRITE sub-transfer: fires on arrival at the other side.
// Separate prefixes so WRITE-path and READ-path handlers can diverge independently.
constexpr char NIXLS_WRITE_CHUNK_PREFIX[]      = "_NIXLS_WCHUNK_";  // fires at receiver
constexpr char NIXLS_READ_CHUNK_PREFIX[]       = "_NIXLS_RCHUNK_";  // fires at initiator
// Slot-freed signal: sent after drain completes to allow the peer to reuse the slot.
constexpr char NIXLS_WRITE_SLOT_FREED_PREFIX[] = "_NIXLS_WFREE_";   // receiver → initiator
constexpr char NIXLS_READ_SLOT_FREED_PREFIX[]  = "_NIXLS_RFREE_";   // initiator → provider

// ---------------------------------------------------------------------------
// Binary serialization helpers
// ---------------------------------------------------------------------------
// Wire messages are plain nixl_blob_t (= std::string) with a fixed prefix
// followed by packed big-endian fields.

inline void packU8(nixl_blob_t& b, uint8_t v) {
    b.push_back(static_cast<char>(v));
}

inline void packU32(nixl_blob_t& b, uint32_t v) {
    b.push_back(static_cast<char>((v >> 24) & 0xff));
    b.push_back(static_cast<char>((v >> 16) & 0xff));
    b.push_back(static_cast<char>((v >>  8) & 0xff));
    b.push_back(static_cast<char>( v        & 0xff));
}

inline void packU64(nixl_blob_t& b, uint64_t v) {
    packU32(b, static_cast<uint32_t>(v >> 32));
    packU32(b, static_cast<uint32_t>(v));
}

inline void packStr(nixl_blob_t& b, const std::string& s) {
    packU32(b, static_cast<uint32_t>(s.size()));
    b.append(s);
}

inline void packDlist(nixl_blob_t& b, const nixl_xfer_dlist_t& dl) {
    packU32(b, static_cast<uint32_t>(dl.getType()));
    packU32(b, static_cast<uint32_t>(dl.descCount()));
    for (int i = 0; i < dl.descCount(); ++i) {
        const nixlBasicDesc& d = dl[i];
        packU64(b, static_cast<uint64_t>(d.addr));
        packU64(b, static_cast<uint64_t>(d.len));
        packU64(b, d.devId);
    }
}

inline uint8_t unpackU8(const nixl_blob_t& b, size_t& o) {
    return static_cast<uint8_t>(b[o++]);
}

inline uint32_t unpackU32(const nixl_blob_t& b, size_t& o) {
    uint32_t v = (static_cast<uint8_t>(b[o])   << 24u) |
                 (static_cast<uint8_t>(b[o+1]) << 16u) |
                 (static_cast<uint8_t>(b[o+2]) <<  8u) |
                  static_cast<uint8_t>(b[o+3]);
    o += 4;
    return v;
}

inline uint64_t unpackU64(const nixl_blob_t& b, size_t& o) {
    return (static_cast<uint64_t>(unpackU32(b, o)) << 32) | unpackU32(b, o);
}

inline std::string unpackStr(const nixl_blob_t& b, size_t& o) {
    uint32_t len = unpackU32(b, o);
    std::string s = b.substr(o, len);
    o += len;
    return s;
}

inline nixl_xfer_dlist_t unpackDlist(const nixl_blob_t& b, size_t& o) {
    auto     type = static_cast<nixl_mem_t>(unpackU32(b, o));
    uint32_t cnt  = unpackU32(b, o);
    nixl_xfer_dlist_t dl(type);
    for (uint32_t i = 0; i < cnt; ++i) {
        auto addr  = static_cast<uintptr_t>(unpackU64(b, o));
        auto len   = static_cast<size_t>   (unpackU64(b, o));
        auto devId =                         unpackU64(b, o);
        dl.addDesc(nixlBasicDesc(addr, len, devId));
    }
    return dl;
}

#endif // NIXL_MARSHAL_UTILS_H
