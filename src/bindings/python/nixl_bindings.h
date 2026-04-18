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
 * @file nixl_bindings.h
 * @brief Shared exception types and helpers for NIXL pybind11 extension modules.
 *
 * Include after pybind11/pybind11.h and nixl.h (provides nixl_status_t).
 */

#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Notification map type shared between _bindings and _service_bindings
// ---------------------------------------------------------------------------
typedef std::map<std::string, std::vector<py::bytes>> nixl_py_notifs_t;

// ---------------------------------------------------------------------------
// Exception classes — all inherit std::runtime_error so callers can catch
// at either the specific type or the generic RuntimeError level.
// ---------------------------------------------------------------------------
class nixlNotPostedError       : public std::runtime_error { public: using runtime_error::runtime_error; };
class nixlInvalidParamError    : public std::runtime_error { public: using runtime_error::runtime_error; };
class nixlBackendError         : public std::runtime_error { public: using runtime_error::runtime_error; };
class nixlNotFoundError        : public std::runtime_error { public: using runtime_error::runtime_error; };
class nixlMismatchError        : public std::runtime_error { public: using runtime_error::runtime_error; };
class nixlNotAllowedError      : public std::runtime_error { public: using runtime_error::runtime_error; };
class nixlRepostActiveError    : public std::runtime_error { public: using runtime_error::runtime_error; };
class nixlUnknownError         : public std::runtime_error { public: using runtime_error::runtime_error; };
class nixlNotSupportedError    : public std::runtime_error { public: using runtime_error::runtime_error; };
class nixlRemoteDisconnectError: public std::runtime_error { public: using runtime_error::runtime_error; };
class nixlCancelledError       : public std::runtime_error { public: using runtime_error::runtime_error; };
class nixlNoTelemetryError     : public std::runtime_error { public: using runtime_error::runtime_error; };

// ---------------------------------------------------------------------------
// throw_nixl_exception — map a nixl_status_t to the matching Python exception
// ---------------------------------------------------------------------------
inline void
throw_nixl_exception(const nixl_status_t &status) {
    switch (status) {
    case NIXL_IN_PROG:
    case NIXL_SUCCESS:
        return;
    case NIXL_ERR_NOT_POSTED:
        throw nixlNotPostedError(nixlEnumStrings::statusStr(status).c_str());
    case NIXL_ERR_INVALID_PARAM:
        throw nixlInvalidParamError(nixlEnumStrings::statusStr(status).c_str());
    case NIXL_ERR_BACKEND:
        throw nixlBackendError(nixlEnumStrings::statusStr(status).c_str());
    case NIXL_ERR_NOT_FOUND:
        throw nixlNotFoundError(nixlEnumStrings::statusStr(status).c_str());
    case NIXL_ERR_MISMATCH:
        throw nixlMismatchError(nixlEnumStrings::statusStr(status).c_str());
    case NIXL_ERR_NOT_ALLOWED:
        throw nixlNotAllowedError(nixlEnumStrings::statusStr(status).c_str());
    case NIXL_ERR_REPOST_ACTIVE:
        throw nixlRepostActiveError(nixlEnumStrings::statusStr(status).c_str());
    case NIXL_ERR_UNKNOWN:
        throw nixlUnknownError(nixlEnumStrings::statusStr(status).c_str());
    case NIXL_ERR_NOT_SUPPORTED:
        throw nixlNotSupportedError(nixlEnumStrings::statusStr(status).c_str());
    case NIXL_ERR_REMOTE_DISCONNECT:
        throw nixlRemoteDisconnectError(nixlEnumStrings::statusStr(status).c_str());
    case NIXL_ERR_CANCELED:
        throw nixlCancelledError(nixlEnumStrings::statusStr(status).c_str());
    case NIXL_ERR_NO_TELEMETRY:
        throw nixlNoTelemetryError(nixlEnumStrings::statusStr(status).c_str());
    default:
        throw std::runtime_error("BAD_STATUS");
    }
}
