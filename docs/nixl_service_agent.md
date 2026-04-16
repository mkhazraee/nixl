# nixlServiceAgent Design Plan

## Overview

nixlAgent is a pure transfer primitive — move bytes from A to B, simply and fast. nixlServiceAgent
adds a service layer on top: compression, staging, and encryption applied transparently during
transfers. The ultimate goal is for users to replace `nixlAgent` with `nixlServiceAgent` at the call
site and gain these capabilities without changing anything else.

To achieve this, nixlServiceAgent inherits `nixlAgent` (IS-A) so the same call structure works
without restructuring user code. The transfer handle `nixlServiceXferReqH` is a composition
type rather than a subtype of `nixlXferReqH` — it tracks multi-step pipeline state rather than
a single transfer. This does not affect Python usage, as handles are returned as typed objects
and duck typing handles dispatch automatically. In C++, a `= delete`d base-type overload
produces a clear compile error at the handle declaration, guiding users to the only update
needed. `DIRECT` mode has the same behavior as the base class, so users who want both plain
and service transfers can toggle per-transfer without switching types.

Service progress is driven by notification callbacks in an event-driven manner — no background
thread is required for correctness; per-backend threads will be added as a performance
optimization (Phase 4) to keep callbacks non-blocking under high concurrency. Services are
organized as **sub-services**. Each sub-service owns a wire protocol and defines an interface
for its backends. The current sub-service is **marshaling**; when the need arises, future
sub-services (encryption, etc.) would each add their own wire protocol and backend interface
without changing the existing ones.


---

## Architecture

### What Marshaling Does

The marshaling sub-service handles transfers where data cannot flow directly from source to final
destination in a single NIXL operation — either because it needs transformation (compression,
encryption) or if user wants to make them contiguous on one or both sides before transfer (staging). In all
these cases the approach is the same: both sender and receiver allocate a dedicated staging buffer.
The sender fills its staging slot (compressing, encrypting, or gathering), sends the slot to the
receiver via a single NIXL_WRITE, and the receiver drains it (decompressing, decrypting, or
scattering) into the final destination.

Because both sides participate — the sender needs a free receiver slot, and the receiver needs
to know what is in the slot before draining — a two-sided wire protocol coordinates the exchange.
This protocol (RTS → CTS → NIXL_WRITE → drain → SLOT_FREED) is identical for all marshal modes.
The staging slot, slot pool management, and wire protocol are owned by nixlServiceAgent. What
varies per mode is only what `fillSlot` and `drainSlot` do — both delegated to a
`nixlMarshalBackend`.

Keeping the protocol and transformation orthogonal means protocol improvements (double buffering,
pipelining) benefit all backends without touching them, and new backends require no changes to the
protocol machinery.

### Marshal Backends

Each non-DIRECT mode maps to a `nixlMarshalBackend` implementation:


| Mode                    | Backend              | fillSlot         | drainSlot          |
| ----------------------- | -------------------- | ---------------- | ------------------ |
| STAGE_BOTH              | staging_backend      | gather-copy      | scatter-copy       |
| COMPRESS                | nvcomp_backend       | compress         | decompress         |
| STAGE_INIT *(future)*   | stage_init_backend   | gather-copy      | identity           |
| STAGE_TARGET *(future)* | stage_target_backend | identity         | scatter-copy       |
| ENC *(future)*          | enc_backend          | encrypt          | decrypt            |
| ENC_COMP *(future)*     | enc_comp_backend     | compress+encrypt | decrypt+decompress |


---

## API

### Service Modes

```cpp
enum class nixl_service_mode_t {
    DIRECT,       // passthrough to base nixlAgent
    STAGE_BOTH,   // marshal sub-service: gather/scatter via staging slot
    COMPRESS,     // marshal sub-service: compress/decompress via staging slot
    // STAGE_INIT, STAGE_TARGET, ENC, ENC_COMP — future marshal backends
};
```

### Configuration

`nixlServiceAgentConfig` extends `nixlAgentConfig` with:

```cpp
struct nixlServiceAgentConfig : nixlAgentConfig {
    nixl_service_mode_t defaultMode = nixl_service_mode_t::DIRECT;

    // COMPRESS: default algorithm to use when none specified per-transfer
    nixl_comp_algo_t    defaultCompAlg = nixl_comp_algo_t::ANS;
    // nixl_enc_algo_t  defaultEncAlg  = ...;  // future: ENC, ENC_COMP modes

    // Per-algorithm staging slot sizes, populated from marshalQuery() before construction.
    // Required for any mode that uses the marshal sub-service; empty is valid for DIRECT-only.
    std::unordered_map<nixl_comp_algo_t, size_t> algSlotSizes;
};
```

`algSlotSizes` is filled by the caller using the output of `marshalQuery()` before constructing
the agent. A DIRECT-only config with an empty `algSlotSizes` is valid. Additional fields are
added as new modes are implemented.

### Per-Transfer Options

`nixlServiceOptArgs` extends `nixl_opt_args_t` with optional per-transfer mode and algorithm
overrides. Resolution order: opt-args → config default → DIRECT.

### Transfer Handle and API Compatibility

`nixlServiceXferReqH` is a standalone type — composition of pipeline stage state and the
individual `nixlXferReqH*` sub-transfer handles it owns internally. It does not inherit
`nixlXferReqH`.

nixlServiceAgent re-declares the five transfer lifecycle methods with `nixlServiceXferReqH`
in place of `nixlXferReqH` (`createXferReq`, `makeXferReq`, `postXferReq`, `getXferStatus`,
`releaseXferReq`) and `= delete`s the base-type overloads. If a user holds the handle as
`nixlXferReqH*` rather than `nixlServiceXferReqH*` (or `auto`), the deleted overload fires a
clear compile error at the declaration site. Methods that require knowledge of the full
pipeline to be meaningful (`estimateXferCost`, `getXferTelemetry`) are also `= delete`d until
service-aware implementations are added.

In Python, `nixlServiceXferReqH` is bound as an independent class with no base:

```cpp
py::class_<nixlServiceXferReqH>(m, "nixlServiceXferReqH")
```

Handles are returned as typed objects from `createXferReq`; Python's duck typing means they
flow through all call sites without type annotations. Replacing `nixlAgent` with
`nixlServiceAgent` at construction is the only change needed in Python.

### marshalQuery

```cpp
namespace nixlService {
    nixl_status_t marshalQuery(MarshalRequirements& reqs);
}

struct MarshalRequirements {
    // Ideal staging slot size per algorithm for the detected GPU architecture.
    // Keys are flexible: the same underlying algorithm at different chunk sizes, or staging
    // at different granularities, can each appear as a distinct entry in this map.
    std::unordered_map<nixl_comp_algo_t, size_t> algSlotSizes;
};
```

`marshalQuery` is a free static function — no agent instance required. It calls each backend
class's static `queryRequirements(reqs)` method, aggregating `algSlotSizes` across all
compiled-in backends. Slot sizes are performance recommendations — the chunk size at which each
backend operates at peak throughput on the detected GPU architecture — not worst-case output
bounds. As an example, `nvcomp_backend` might return ~256 MB for ANS (optimal for FP16
KV-cache entropy coding) and ~16–64 MB for Bitcomp (FP4 weights, where finer granularity
helps pipeline overlap). `staging_backend` returns no entries in `algSlotSizes`; its slot size
is user-configured (e.g. a `stageSlotSize` field in config) since it is not
architecture-dependent. If the wire protocol employs double buffering, `marshalQuery` reports
2× the underlying size so the caller allocates the correct staging memory without needing to
know the buffering strategy. Returns `NIXL_SUCCESS` on success or an error if any backend
query fails (e.g. no CUDA device present for nvCOMP).

### Staging Memory Registration

```cpp
nixl_status_t registerServiceMem  (const nixl_reg_dlist_t& descs, nixl_service_mode_t use, ...);
nixl_status_t deregisterServiceMem(const nixl_reg_dlist_t& descs, nixl_service_mode_t use, ...);
```

Staging memory is separate from the application's regular NIXL-registered buffers and must be
registered explicitly before metadata exchange. `registerServiceMem` is forwarded to the marshal
backend for `use`; the backend handles transport registration and any library-specific setup.

Registration does not take an algorithm parameter. Slot granularity is
`min(algSlotSizes.values())` from `marshalQuery()`, so all algorithms share the same staging
memory: `numSlots = floor(total_size / min_slot_size)`; less than one slot returns
`NIXL_ERR_INVALID_PARAM`. Wire messages carry only a slot number; both sides compute the
address as `base + slot_num * slot_size`, where `base` is the start of the registered service
buffer.

nixlServiceAgent extends `getLocalMD`/`loadRemoteMD` to append and parse service metadata using
the same `nixlSerDes` tag-value format used throughout nixl — consistent with how backend
engines attach per-memory metadata via `getPublicData()`. Per registered service buffer it
serializes: service agent name, slot size, and slot count as additional tagged fields. Because
the staging buffer is registered with nixl transport backends (e.g. UCX), the RDMA keys (rkeys)
and base address are already included in the standard nixl memory descriptor metadata — the
service metadata layer only adds the slot layout on top. This metadata must be re-exchanged
after any `registerServiceMem`/`deregisterServiceMem` call, consistent with the base class.
Both peers must register before metadata exchange.

---

## Wire Protocol

Service protocol messages (RTS, CTS, slot-freed, final completion) are `genNotif` calls with
`_NIXLS_*`-prefixed strings, routed by callback to the per-backend queue. Intermediate
per-chunk signals are not separate `genNotif` calls — they are notification strings attached
to each NIXL_WRITE xferReq; the transport delivers them to the receiver automatically when
the write lands. RDMA keys and base addresses are known from metadata exchange; wire messages
carry slot indices only. Slot-freed carries the specific index, enabling non-consecutive reuse
and double buffering without physically adjacent slots.

### WRITE

RTS carries: `transfer_id`, algo, slot_size, intermediate notification prefix, final_notif_id
(delivered to the user when all drains complete), and final destination descriptors. The
receiver's callback handler allocates a slot and replies with CTS. The initiator pipelines
chunks: fillSlot (async) → checkFill(slot_addr) → NIXL_WRITE with intermediate_prefix+N
attached. The receiver's callback handler sees the intermediate notification, drains the slot
(async), checkDrain(slot_addr), then sends slot-freed. The slot_freed arriving at the initiator
confirms the write completed and frees both slots for the next chunk. When all drains complete,
the receiver delivers final_notif_id to the user.

```
Initiator                                              Receiver
---------                                              --------
genNotif(RTS{transfer_id, algo, slot_size,  -->
             intermediate_prefix,
             final_notif_id, final_dst})
                                                       [callback: alloc slot R]
                                   <-- genNotif(CTS{transfer_id, slot_R})

-- per chunk N --
fillSlot(src_N → slot_I)             [async GPU]
checkFill(slot_I_addr) == DONE
postXferReq(NIXL_WRITE slot_I → slot_R,  -->
            notif=intermediate_prefix+N)
                                                       [callback: intermediate notif arrived]
                                                       drainSlot(slot_R → dst_N)  [async GPU]
                                                       checkDrain(slot_R_addr) == DONE
                                   <-- genNotif(slot_freed{slot_R})
[callback: slot freed → reuse both slots; start next chunk]
-- end per chunk --

[callback: last slot_freed → mark initiator side COMPLETE]

                                                       [all chunks drained]
                                                       genNotif(final_notif_id)
```

### READ

READ_REQ carries: `transfer_id`, initiator's pre-allocated slot indices, algo, slot_size,
intermediate notification prefix, and source descriptors. The remote fills each slot via
NIXL_WRITE with the intermediate notification attached. The initiator's callback handler drains
on arrival, then sends slot-freed so the remote can reuse without waiting. When all drains are
done, the initiator sends a final completion notification to the remote.

```
Initiator                                              Remote
---------                                              ------
[alloc local slots I]
genNotif(READ_REQ{transfer_id,              -->
                  slot_indices, algo,
                  slot_size, intermediate_prefix,
                  src_descs})

                                                       [callback: start fills per chunk]
                                                       fillSlot(src_N → slot_I) [async GPU]
                                                       checkFill(slot_I_addr) == DONE
                                                       postXferReq(NIXL_WRITE → slot_I,
                                   <--                 notif=intermediate_prefix+N)

[callback: intermediate notif arrived]
drainSlot(slot_I → final_dst_N)    [async GPU]
checkDrain(slot_I_addr) == DONE
genNotif(slot_freed{slot_I})       -->
                                                       [slot_I available for next chunk]

[all drains complete]
genNotif(final_notif_to_remote)    -->
mark initiator side COMPLETE
```

---

### Marshal Backends API

`nixlMarshalBackend` defines what each implementation must provide:

- **Capabilities:** `getSupportedMems()` — memory types accepted for staging;
`static queryRequirements(MarshalRequirements&)` — each backend class exposes a static
method that reports its ideal slot size for the detected GPU architecture;
`marshalQuery` aggregates across all backend classes by calling these static methods
- **Resource management:** `registerStagingMem(descs, agent)`,
`deregisterStagingMem(descs, agent)` — each backend independently registers staging buffers
with the underlying nixlAgent transport (making them RDMA-accessible) and with its own library
if needed (e.g. nvCOMP workspace setup)
- **Transformation:** `fillSlot(src, slot_addr)`, `drainSlot(slot_addr, size, dst)` — submit
the async GPU operation; `checkFill(slot_addr)` / `checkDrain(slot_addr)` poll for completion
using the slot address as the unique key (a slot is owned by at most one operation at a time)

`DIRECT` mode involves no marshal backend — it delegates entirely to the inherited base class.

---

## Progress Handling

### Callback Injection

nixlServiceAgent uses a static helper to populate `notifCallbacks` in the config before the
base constructor runs — one entry per backend sub-prefix (e.g. `"_NIXLS_STAGE_"`,
`"_NIXLS_COMP_"`). This is the same injection pattern used to avoid any ordering issue between
queue construction and base initialization:

```cpp
nixlServiceAgent::nixlServiceAgent(const std::string& name, nixlServiceAgentConfig cfg)
    : nixlAgent(name, injectCallbacks(cfg, stagingQueue_, compQueue_, ...))
```

The base agent routes matching notifications to the registered callback at arrival; all others
reach the user's `getNotifs()` unchanged. The callback runs the state machine directly — no
intermediate queue is needed for Phase 2/3. No backend registers its own callback —
nixlServiceAgent owns all `_NIXLS_*` routing.

### State Machine and checkXfer

The callback is the sole progress mechanism — no polling or progress thread is needed for
correctness. Every incoming wire message (CTS, slot-freed, READ_REQ, intermediate notification)
triggers the next protocol action directly in the callback: submit GPU op, poll
`checkFill`/`checkDrain` synchronously (fast for GPU ops), post the next NIXL_WRITE or send
the next wire reply. The transfer handle's completion flag is set by the callback when all
work is done.

`getXferStatus(nixlServiceXferReqH*)` is therefore a pure status query — it reads the flag
the callback set. No progress is driven by calling it.

Phase 4 moves GPU op polling out of the callback and into a per-backend thread, making
callbacks non-blocking. This is a performance optimization — if callbacks become a bottleneck
(e.g. many concurrent transfers), one thread per backend handles the GPU completion polling
asynchronously. Until then, synchronous polling in the callback is sufficient.

---

## File Structure

### New: `src/nixlService/`


| File                    | Purpose                                                                                                                |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `nixl_service_types.h`  | `nixl_service_mode_t`, `nixl_comp_algo_t`, handle types, config, opt args, wire message structs, `MarshalRequirements` |
| `nixl_service.h`        | `nixlServiceAgent` class declaration; `nixlService::marshalQuery`                                                      |
| `nixl_service_data.h`   | Pimpl `nixlServiceAgentData`: slot pools per peer, pending transfer map                                               |
| `nixl_service.cpp`      | nixlServiceAgent implementation and wire protocol state machine                                                        |
| `marshal_backend.h`     | `nixlMarshalBackend` abstract interface                                                                                |
| `staging_backend.h/cpp` | STAGE_BOTH: fill/drain via `cudaMemcpyAsync` gather/scatter                                                            |
| `nvcomp_backend.h/cpp`  | COMPRESS: fill/drain via nvCOMP ANS compress/decompress                                                                |
| `meson.build`           | Conditional on `build_nixl_service`                                                                                    |


### Modified Files


| File                 | Change                                                                    |
| -------------------- | ------------------------------------------------------------------------- |
| `src/api/cpp/nixl.h` | Add `virtual ~nixlAgent()`                                                |
| `meson_options.txt`  | Add `nvcomp_path` (string), `build_nixl_service` (boolean, default false) |
| `src/meson.build`    | Add conditional `subdir('nixlService')`                                   |


---

## Implementation Phases

### Phase 0: Scaffolding

Add `virtual ~nixlAgent()`, build options, and `src/nixlService/` directory stub.
**Milestone:** clean build with no regressions.

### Phase 1: Types and Headers

Define `nixl_service_types.h`, `marshal_backend.h`, `nixl_service.h`, `nixl_service_data.h`.
Wire message structs (RTS, CTS, READ_REQ, slot-freed) with their fields; `nixlMarshalBackend`
interface including `fillSlot`/`drainSlot`/`checkFill`/`checkDrain`.
**Milestone:** all headers compile; `nixlMarshalBackend` interface ready for implementations.

### Phase 2: staging_backend + STAGE_BOTH

Implement `staging_backend` via `cudaMemcpyAsync`. Implement DIRECT passthrough and full
STAGE_BOTH WRITE and READ state machines with the pipelined, slot-freed-based wire protocol.
State machine is driven entirely by callbacks; GPU op polling is synchronous in the callback.
`checkXfer` on service handles is a pure status query.
**Milestone:** two-process WRITE and READ with non-contiguous descriptors verified; nixlAgent
tests pass in DIRECT mode.

### Phase 3: nvcomp_backend + COMPRESS

Implement `nvcomp_backend` via nvCOMP ANS. Wire protocol unchanged from Phase 2; only
`fillSlot`/`drainSlot` and their async completion differ. Implement `marshalQuery`
(`staging_backend` returns no entries; `nvcomp_backend` populates `algSlotSizes` for ANS).

Two implementation constraints to observe:
- **Workspace isolation:** nvCOMP requires a temporary workspace per active transfer
  (`getCompressTempSize`). This workspace must never be shared across concurrent transfers —
  two concurrent transfers sharing a workspace will silently corrupt each other's intermediate
  state. Each `nixlServiceXferReqH` owns its workspace for the lifetime of the transfer.
- **GPUNETIO stream ordering:** `nvcomp_backend` uses a compression CUDA stream and a transfer
  stream ordered via `cudaEventRecord`/`cudaStreamWaitEvent`. This ordering must be explicitly
  validated against the GPUNETIO backend, which has its own internal stream model that may
  interact with external stream dependencies.

**Milestone:** WRITE and READ compress/decompress verified at multiple transfer sizes.

### Phase 4: Per-Backend Thread (Optional Performance)

Move GPU op polling (`checkFill`/`checkDrain`) out of callbacks and into a per-backend thread,
making callbacks non-blocking. Correctness is unchanged — this only matters when callback
latency becomes a bottleneck under high concurrency.
**Milestone:** throughput under concurrent transfers matches Phase 2/3; callbacks return
immediately.

### Phase 5: Python Bindings

Bind `nixlServiceAgent`, config, opt args, handle type, and `marshalQuery`.
**Milestone:** DIRECT, STAGE_BOTH, and COMPRESS end-to-end in Python.

---

## Out of Scope

- STAGE_INIT, STAGE_TARGET, ENC, ENC_COMP — future `nixlMarshalBackend` implementations
- Multiple compression algorithms beyond ANS
- CPU compression backend (DRAM staging via `getSupportedMems`)

