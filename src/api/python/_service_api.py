# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Service-layer Python API — diff from nixl_agent.

nixl_service_agent inherits all of nixl_agent's methods.  Only what differs:
  - _build_agent       override: creates nixlServiceAgent instead of nixlAgent
  - initialize_xfer   override: adds optional mode / comp_alg parameters
  - register_service_memory / deregister_service_memory  (service-only)
  - marshal_query     (service-only)
"""

from typing import Any, Optional

from . import _bindings as nixlBind  # type: ignore
from . import _service_bindings as svcBind  # type: ignore
from ._api import nixl_agent, nixl_agent_config
from .logging import get_logger

logger = get_logger(__name__)

_MODE_MAP = {
    "DIRECT": svcBind.DIRECT,
    "STAGE_BOTH": svcBind.STAGE_BOTH,
    "COMPRESS": svcBind.COMPRESS,
}


class nixl_service_xfer_handle:
    """Opaque wrapper for a nixlServiceXferReqH* transfer handle."""

    __slots__ = ("_handle", "_agent", "_released")

    def __init__(self, agent: "nixl_service_agent", value: int):
        self._handle = int(value)
        self._agent: Any = (
            agent  # Any: .agent attr comes from pybind11, type unresolvable
        )
        self._released = False

    def __repr__(self) -> str:
        return (
            f"nixl_service_xfer_handle(0x{self._handle:x}, released={self._released})"
        )

    def release(self) -> None:
        if not self._released:
            self._agent.agent.releaseXferReq(self._handle)
            self._released = True

    def __del__(self) -> None:
        if not self._released:
            try:
                self._agent.agent.releaseXferReq(self._handle)
            except Exception as e:
                try:
                    logger.error(
                        "Failed to release nixl_service_xfer_handle 0x%x: %s",
                        self._handle,
                        e,
                    )
                except Exception:
                    pass


class nixl_service_agent_config(nixl_agent_config):
    """
    Configuration for nixl_service_agent.

    Inherits all nixl_agent_config fields and adds service-specific ones.
    """

    def __init__(
        self,
        enable_prog_thread: bool = True,
        enable_listen_thread: bool = False,
        listen_port: int = 0,
        capture_telemetry: bool = False,
        backends: list[str] = ["UCX"],
        default_mode: str = "DIRECT",
        default_comp_alg: Optional[str] = None,
    ):
        super().__init__(
            enable_prog_thread=enable_prog_thread,
            enable_listen_thread=enable_listen_thread,
            listen_port=listen_port,
            capture_telemetry=capture_telemetry,
            backends=backends,
        )
        self.default_mode = default_mode
        self.default_comp_alg = default_comp_alg


class nixl_service_agent(nixl_agent):
    """
    Drop-in replacement for nixl_agent with optional staging/compression services.

    Inherits all nixl_agent methods unchanged.  DIRECT mode behaviour is
    identical to nixl_agent.  Additional modes route transfers through staging
    buffers registered via register_service_memory().
    """

    def __init__(
        self,
        agent_name: str,
        nixl_conf: Optional[nixl_service_agent_config] = None,
        instantiate_all: bool = False,
    ):
        if not nixl_conf:
            nixl_conf = nixl_service_agent_config()
        super().__init__(agent_name, nixl_conf, instantiate_all)

    def _build_agent(self, agent_name: str, nixl_conf: nixl_agent_config) -> None:
        """Override: translate service config and instantiate a nixlServiceAgent."""
        assert isinstance(nixl_conf, nixl_service_agent_config)
        cfg = svcBind.nixlServiceAgentConfig()
        cfg.useProgThread = nixl_conf.enable_pthread
        cfg.useListenThread = nixl_conf.enable_listen
        cfg.listenPort = nixl_conf.port
        cfg.syncMode = (
            nixlBind.NIXL_THREAD_SYNC_STRICT
            if nixl_conf.enable_listen
            else nixlBind.NIXL_THREAD_SYNC_NONE
        )
        cfg.captureTelemetry = nixl_conf.capture_telemetry
        cfg.defaultMode = _MODE_MAP[nixl_conf.default_mode]
        if nixl_conf.default_comp_alg is not None:
            cfg.defaultCompAlg = nixl_conf.default_comp_alg
        self.agent = svcBind.nixlServiceAgent(agent_name, cfg)

    # -----------------------------------------------------------------------
    # Override: adds optional mode / comp_alg, returns nixl_service_xfer_handle
    # -----------------------------------------------------------------------

    def initialize_xfer(  # type: ignore[override]
        self,
        operation: str,
        local_descs,
        remote_descs,
        remote_agent: str,
        notif_msg: bytes = b"",
        backends: list[str] = [],
        mode: Optional[str] = None,
        comp_alg: Optional[str] = None,
    ) -> nixl_service_xfer_handle:
        op = self.nixl_ops[operation]  # type: ignore[has-type]
        handles = [self.backends[b] for b in backends if b in self.backends]
        notif = notif_msg.decode("utf-8") if isinstance(notif_msg, bytes) else notif_msg
        mode_v = _MODE_MAP[mode] if mode is not None else None
        raw = self.agent.createXferReq(
            op,
            local_descs,
            remote_descs,
            remote_agent,
            notif,
            handles,
            mode_v,
            comp_alg,
        )
        return nixl_service_xfer_handle(self, raw)

    # -----------------------------------------------------------------------
    # Service-only additions
    # -----------------------------------------------------------------------

    def register_service_memory(
        self,
        reg_list,
        mode: str = "STAGE_BOTH",
        backends: list[str] = [],
    ) -> None:
        handles = [self.backends[b] for b in backends if b in self.backends]
        self.agent.registerServiceMem(reg_list, _MODE_MAP[mode], handles)

    def deregister_service_memory(
        self,
        reg_list,
        mode: str = "STAGE_BOTH",
        backends: list[str] = [],
    ) -> None:
        handles = [self.backends[b] for b in backends if b in self.backends]
        self.agent.deregisterServiceMem(reg_list, _MODE_MAP[mode], handles)

    def marshal_query(self) -> svcBind.MarshalRequirements:
        return svcBind.marshalQuery()
