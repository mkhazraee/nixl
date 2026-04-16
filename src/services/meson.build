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

# services — compression/staging service layer on top of nixlAgent.
# Enabled via: -Dbuild_nixl_service=true
# For COMPRESS mode (Phase 3), also set: -Dnvcomp_path=/path/to/nvcomp

nixl_service_inc = include_directories('.')

nixl_service_src = ['nixl_service.cpp',
                    'marshal/marshal_protocol.cpp',
                    'marshal/staging/staging_backend.cpp']
nixl_service_deps = [nixl_dep, nixl_infra, nixl_common_dep]
nixl_service_cpp_args = []

# Phase 2: staging_backend uses cudaMemcpyAsync when CUDA is available,
# falls back to memcpy for DRAM-to-DRAM when CUDA is not present.
if cuda_dep.found()
    nixl_service_deps += [cuda_dep]
    nixl_service_cpp_args += ['-DNIXL_SERVICE_HAVE_CUDA']
else
    message('services: CUDA not found — STAGE_BOTH will use memcpy (DRAM only)')
endif

# Phase 3: add nvCOMP dependency when nvcomp_path is set
nvcomp_path = get_option('nvcomp_path')
if nvcomp_path != ''
    nvcomp_inc = include_directories(nvcomp_path / 'include')
    nvcomp_lib = cpp.find_library('nvcomp',
        dirs: nvcomp_path / 'lib',
        required: true)
    nixl_service_deps += [declare_dependency(
        include_directories: nvcomp_inc,
        dependencies: nvcomp_lib)]
    # TODO (Phase 3): add nvcomp_backend.cpp to nixl_service_src
endif

nixl_service_lib = library(
    'nixl_service',
    nixl_service_src,
    cpp_args: nixl_service_cpp_args,
    include_directories: [nixl_inc_dirs, utils_inc_dirs, nixl_service_inc],
    dependencies: nixl_service_deps,
    install: true)

nixl_service_dep = declare_dependency(
    include_directories: nixl_service_inc,
    link_with: nixl_service_lib,
    dependencies: nixl_dep)

if get_option('install_headers')
    install_headers(
        'nixl_service.h',
        'nixl_service_types.h',
        'marshal/marshal_backend.h',
        subdir: 'nixl')
endif
