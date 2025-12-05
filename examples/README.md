# NIXL Examples

This directory contains runnable examples across Python, C++, and Rust.

 The `examples/python/basic_two_peers.py` is a good starting point to learn the basic usage of NIXL end-to-end.

 What it demonstrates (fundamentals):
 - Registration best practice: fewer, larger registrations reduce kernel calls and internal lookups
 - Creating descriptors used for transfers in various modes, from tensor, python tuples, numpy
 - Creating transfers in different modes:
  - Prepped flow (prep_xfer_dlist + make_prepped_xfer): when blocks are known in advance, prepare once and choose indices/mappings at runtime
  - Combined flow (initialize_xfer): NIXL prepares and creates the transfer handle in one step; useful when there is no fixed notion of blocks
 - Application-driven ordering vs. parallelism (posting groups and waiting to enforce order)
 - Reposting transfer handles after completion
 - Notification checks, data verification, and explicit teardown at the end


## Python examples
- `examples/python/basic_two_peers.py`: Dual-node example with parallel READs, parallel WRITEs, and READ reposts.
- `examples/python/nixl_api_example.py`: Broader API tour including repeated postings and combined/prepped flows.
- `examples/python/nixl_gds_example.py`: GPU Direct Storage flow (host memory to files and back).
- `examples/python/partial_md_example.py`: Partial metadata exchange and retries before issuing transfers.
- `examples/python/query_mem_example.py`: Query memory/storage info for registered regions.
- `examples/python/telemetry_reader.py`: Reading transfer telemetry from the agent.
- `examples/python/remote_storage_example/`: Peer-to-peer storage pipeline (client-server) with diagrams and utilities.

## C++ examples
- `examples/cpp/nixl_example.cpp`: Basic agent usage and transfers.
- `examples/cpp/nixl_etcd_example.cpp`: Metadata exchange example using etcd.
- `examples/cpp/telemetry_reader.cpp`: Telemetry reader for transfers.

## Rust examples
- `examples/rust/src/single_process_example.rs`: Single-process transfer example.
