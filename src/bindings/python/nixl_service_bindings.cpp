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
 * @file nixl_service_bindings.cpp
 * @brief pybind11 bindings for nixlServiceAgent and related types.
 *
 * nixlServiceAgent inherits nixlAgent in pybind11 — all nixlAgent methods
 * are available without repetition. Only the constructor, service-specific
 * methods, and overridden methods (different handle type) are bound here.
 *
 * _bindings must be imported before this module so that nixlAgent's type
 * is registered in the pybind11 global type registry.
 */

#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nixl.h"
#include "nixl_service.h"
#include "nixl_bindings.h"

PYBIND11_MODULE(_service_bindings, m) {

    // Ensure _bindings is loaded so nixlAgent's type is in the registry
    py::module_::import("nixl._bindings");

    m.doc() = "pybind11 bindings for nixlServiceAgent — service layer (staging, compression) "
              "on top of nixlAgent. Inherits all nixlAgent methods; only service additions and "
              "overrides (different handle type) are defined here.";

    // Exception translators are already registered globally by _bindings

    // -----------------------------------------------------------------------
    // nixl_service_mode_t
    // -----------------------------------------------------------------------
    py::enum_<nixl_service_mode_t>(m, "nixl_service_mode_t")
        .value("DIRECT",     nixl_service_mode_t::DIRECT)
        .value("STAGE_BOTH", nixl_service_mode_t::STAGE_BOTH)
        .value("COMPRESS",   nixl_service_mode_t::COMPRESS)
        .export_values();

    // -----------------------------------------------------------------------
    // nixlServiceAgentConfig
    //
    // Declared as a pybind11 subclass of nixlAgentConfig
    // -----------------------------------------------------------------------
    py::class_<nixlServiceAgentConfig, nixlAgentConfig>(m, "nixlServiceAgentConfig")
        .def(py::init<>())
        .def_readwrite("defaultMode",    &nixlServiceAgentConfig::defaultMode)
        .def_readwrite("defaultCompAlg", &nixlServiceAgentConfig::defaultCompAlg);

    // -----------------------------------------------------------------------
    // MarshalSlotReq / MarshalRequirements
    // -----------------------------------------------------------------------
    py::class_<MarshalSlotReq>(m, "MarshalSlotReq")
        .def(py::init<>())
        .def_readwrite("mode",     &MarshalSlotReq::mode)
        .def_readwrite("subKey",   &MarshalSlotReq::subKey)
        .def_readwrite("slotSize", &MarshalSlotReq::slotSize);

    py::class_<MarshalRequirements>(m, "MarshalRequirements")
        .def(py::init<>())
        .def_readwrite("entries", &MarshalRequirements::entries);

    m.def("marshalQuery",
          []() -> MarshalRequirements {
              MarshalRequirements reqs;
              throw_nixl_exception(nixlService::marshalQuery(reqs));
              return reqs;
          });

    // -----------------------------------------------------------------------
    // nixlServiceAgent
    //
    // Declared as a pybind11 subclass of nixlAgent so all inherited methods
    // (getAvailPlugins, createBackend, registerMem, getLocalMD, getNotifs,
    // etc.) are available without repetition.  Only the constructor, the two
    // service-specific memory methods, and the transfer lifecycle methods that
    // use a different handle type (nixlServiceXferReqH* vs nixlXferReqH*) are
    // defined here.
    // -----------------------------------------------------------------------
    py::class_<nixlServiceAgent, nixlAgent>(m, "nixlServiceAgent")
        .def(py::init<std::string, nixlServiceAgentConfig>())

        // --- Staging memory (service-only) ---
        .def("registerServiceMem",
             [](nixlServiceAgent &agent,
                nixl_reg_dlist_t descs,
                nixl_service_mode_t mode,
                const std::vector<uintptr_t> &backends) -> nixl_status_t {
                 nixl_opt_args_t extra_params;
                 for (uintptr_t b : backends) extra_params.backends.push_back((nixlBackendH *)b);
                 nixl_status_t ret = agent.registerServiceMem(descs, mode, &extra_params);
                 throw_nixl_exception(ret);
                 return ret;
             },
             py::arg("descs"),
             py::arg("mode"),
             py::arg("backends") = std::vector<uintptr_t>({}),
             py::call_guard<py::gil_scoped_release>())
        .def("deregisterServiceMem",
             [](nixlServiceAgent &agent,
                nixl_reg_dlist_t descs,
                nixl_service_mode_t mode,
                const std::vector<uintptr_t> &backends) -> nixl_status_t {
                 nixl_opt_args_t extra_params;
                 for (uintptr_t b : backends) extra_params.backends.push_back((nixlBackendH *)b);
                 nixl_status_t ret = agent.deregisterServiceMem(descs, mode, &extra_params);
                 throw_nixl_exception(ret);
                 return ret;
             },
             py::arg("descs"),
             py::arg("mode"),
             py::arg("backends") = std::vector<uintptr_t>({}),
             py::call_guard<py::gil_scoped_release>())

        // --- Transfer lifecycle (handle = nixlServiceXferReqH* as uintptr_t) ---
        // These shadow the nixlAgent methods that use nixlXferReqH*.
        .def("createXferReq",
             [](nixlServiceAgent &agent,
                const nixl_xfer_op_t &operation,
                const nixl_xfer_dlist_t &local_descs,
                const nixl_xfer_dlist_t &remote_descs,
                const std::string &remote_agent,
                const std::string &notif_msg,
                const std::vector<uintptr_t> &backends,
                std::optional<nixl_service_mode_t> mode,
                std::optional<std::string> comp_alg) -> uintptr_t {
                 nixlServiceXferReqH *handle = nullptr;
                 nixl_service_opt_args_t extra_params;
                 for (uintptr_t b : backends)
                     extra_params.backends.push_back((nixlBackendH *)b);
                 if (!notif_msg.empty())
                     extra_params.notif = notif_msg;
                 if (mode.has_value())
                     extra_params.mode = *mode;
                 if (comp_alg.has_value())
                     extra_params.compAlg = *comp_alg;
                 throw_nixl_exception(agent.createXferReq(
                     operation, local_descs, remote_descs, remote_agent, handle, &extra_params));
                 return (uintptr_t)handle;
             },
             py::arg("operation"),
             py::arg("local_descs"),
             py::arg("remote_descs"),
             py::arg("remote_agent"),
             py::arg("notif_msg")  = "",
             py::arg("backends")   = std::vector<uintptr_t>({}),
             py::arg("mode")       = std::nullopt,
             py::arg("comp_alg")   = std::nullopt,
             py::call_guard<py::gil_scoped_release>())
        .def("makeXferReq",
             [](nixlServiceAgent &agent,
                const nixl_xfer_op_t &operation,
                uintptr_t local_side,
                const std::vector<int> &local_indices,
                uintptr_t remote_side,
                const std::vector<int> &remote_indices,
                const std::string &notif_msg,
                const std::vector<uintptr_t> &backends,
                std::optional<nixl_service_mode_t> mode,
                std::optional<std::string> comp_alg) -> uintptr_t {
                 nixlServiceXferReqH *handle = nullptr;
                 nixl_service_opt_args_t extra_params;
                 for (uintptr_t b : backends)
                     extra_params.backends.push_back((nixlBackendH *)b);
                 if (!notif_msg.empty())
                     extra_params.notif = notif_msg;
                 if (mode.has_value())
                     extra_params.mode = *mode;
                 if (comp_alg.has_value())
                     extra_params.compAlg = *comp_alg;
                 throw_nixl_exception(agent.makeXferReq(
                     operation,
                     (nixlDlistH *)local_side,  local_indices,
                     (nixlDlistH *)remote_side, remote_indices,
                     handle, &extra_params));
                 return (uintptr_t)handle;
             },
             py::arg("operation"),
             py::arg("local_side"),
             py::arg("local_indices"),
             py::arg("remote_side"),
             py::arg("remote_indices"),
             py::arg("notif_msg")  = "",
             py::arg("backends")   = std::vector<uintptr_t>({}),
             py::arg("mode")       = std::nullopt,
             py::arg("comp_alg")   = std::nullopt,
             py::call_guard<py::gil_scoped_release>())
        .def("postXferReq",
             [](nixlServiceAgent &agent,
                uintptr_t handle,
                const std::string & /*notif_msg*/) -> nixl_status_t {
                 // notif_msg is set at createXferReq time; accepted here for
                 // API compatibility with nixlAgent.postXferReq(handle, msg).
                 nixl_status_t ret = agent.postXferReq((nixlServiceXferReqH *)handle);
                 if (ret < NIXL_SUCCESS) throw_nixl_exception(ret);
                 return ret;
             },
             py::arg("handle"),
             py::arg("notif_msg") = "",
             py::call_guard<py::gil_scoped_release>())
        .def("getXferStatus",
             [](nixlServiceAgent &agent, uintptr_t handle) -> nixl_status_t {
                 return agent.getXferStatus((nixlServiceXferReqH *)handle);
             },
             py::call_guard<py::gil_scoped_release>())
        .def("releaseXferReq",
             [](nixlServiceAgent &agent, uintptr_t handle) -> nixl_status_t {
                 nixl_status_t ret = agent.releaseXferReq((nixlServiceXferReqH *)handle);
                 throw_nixl_exception(ret);
                 return ret;
             });
}
