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
needed. Same goes for the change from `nixl_service_opt_args_t` in place of `nixl_opt_args_t`.
`DIRECT` mode has the same behavior as the base class, so users who want both plain
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
};
```

Additional fields are added as new modes are implemented.

### Per-Transfer Options

`nixl_service_opt_args_t` extends `nixl_opt_args_t` with optional per-transfer mode and algorithm
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

In Python, `nixlServiceXferReqH` is bound as an independent class with no base, and
`nixl_service_opt_args_t` is bound inheriting from `nixl_opt_args_t`:

```cpp
py::class_<nixlServiceXferReqH>(m, "nixlServiceXferReqH")
py::class_<nixl_service_opt_args_t, nixl_opt_args_t>(m, "nixlServiceOptArgs")
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

`marshalQuery` is a free function — no agent instance required. It aggregates `algSlotSizes`
by instantiating each compiled-in backend and calling its virtual `queryRequirements(reqs)` method. Slot sizes are performance recommendations — the chunk size at which each
backend operates at peak throughput on the detected GPU architecture — not worst-case output
bounds. As an example, `nvcomp_backend` might return ~256 MB for ANS (optimal for FP16
KV-cache entropy coding) and ~16–64 MB for Bitcomp (FP4 weights, where finer granularity
helps pipeline overlap). `staging_backend` returns no entries in `algSlotSizes`; an empty result means
the whole registered buffer is treated as a single slot, which is not architecture-dependent. If the wire protocol employs double buffering, `marshalQuery` reports
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

Registration does not take an algorithm parameter. Slot granularity is derived from
`queryRequirements()`: the minimum non-zero value in `algSlotSizes`, or the full buffer size if
the map is empty (staging_backend). `numSlots = ceil(total_size / slot_size)`; less than one
slot returns `NIXL_ERR_INVALID_PARAM`. Wire messages carry only a slot number; both sides compute the
address as `base + slot_num * slot_size`, where `base` is the start of the registered service
buffer.

nixlServiceAgent extends `getLocalMD`/`loadRemoteMD` to include service staging pool layout
alongside the standard nixlAgent metadata. Per registered service buffer it exchanges: base
address, slot size, slot count, device ID, and memory type. The base nixlAgent blob already
includes RDMA keys for the staging buffer; the service layer adds only the slot layout on top.
Wire messages carry slot indices only; both peers compute addresses as `base + idx * slotSize`.
This metadata must be re-exchanged after any `registerServiceMem`/`deregisterServiceMem` call.
Both peers must register staging memory before exchanging metadata.

---

## Wire Protocol

Service protocol messages (RTS, CTS, slot-freed) are `genNotif` calls with `_NIXLS_*`-prefixed
strings, routed by callback to the per-backend queue. Intermediate per-chunk signals are not
separate `genNotif` calls — they are notification strings attached to each NIXL_WRITE xferReq;
the transport delivers them to the receiver automatically when the write lands. The final
user-visible notification is a plain `genNotif` with the user's own `notif_msg` string (not
`_NIXLS_*`-prefixed), delivered locally after the last chunk is processed. RDMA keys and base addresses are known from metadata exchange; wire messages
carry slot indices only. Slot-freed carries the specific index, enabling non-consecutive reuse
and double buffering without physically adjacent slots.

WRITE and READ paths use distinct prefixes for chunk notifications and slot-freed signals
(`_NIXLS_WCHUNK_` / `_NIXLS_RCHUNK_`, `_NIXLS_WFREE_` / `_NIXLS_RFREE_`) so they route to
separate handlers and can diverge independently as new behaviors are added per direction.

### WRITE

RTS carries: `transfer_id`, `mode`, `[algo]`, `slot_size`, `notif_msg`, and final destination
descriptors. `mode` tells the receiver which backend handles the slot (staging, compression, …);
`[algo]` is present only when the mode requires an algorithm selector; `notif_msg` is the
user-specified notification string from `opt.notif` — delivered to the receiver's user after
the last chunk is drained. The receiver allocates a slot and replies with CTS. The initiator pipelines
chunks: fillSlot (async) → checkFill(slot_addr) → NIXL_WRITE with a chunk notification
`_NIXLS_WCHUNK_{transfer_id, N}` attached (`transfer_id` is the whole-transfer ID; `N` is the
per-transfer chunk sequence number; `isLast` is 1 on the final chunk, 0 otherwise — no total
count is sent in RTS, so the receiver learns the end from this flag). As each chunk lands, the
receiver drains the slot and immediately sends WFREE back to the initiator, allowing slot reuse
(backpressure and double buffering). WFREE also carries `isLast`. On the last chunk the receiver
additionally delivers `notif_msg` to its own user. When the last WFREE arrives at the initiator,
the initiator marks the transfer complete.

```
Initiator                                              Receiver
---------                                              --------
genNotif(RTS{transfer_id, mode, [algo],     -->
             slot_size, notif_msg, final_dst})
                                                       [callback: alloc slot R]
                                   <-- genNotif(CTS{transfer_id, slot_R})

-- per chunk N --
fillSlot(src_N → slot_I)             [async GPU]
checkFill(slot_I_addr) == DONE
postXferReq(NIXL_WRITE slot_I → slot_R,  -->
            notif=_NIXLS_WCHUNK_{transfer_id, N, isLast})
                                                       [callback: chunk notif arrived]
                                                       drainSlot(slot_R → dst_N)  [async GPU]
                                                       checkDrain(slot_R_addr) == DONE
                                   <-- genNotif(_NIXLS_WFREE_{transfer_id, slot_R, isLast})
[callback: free slot_I (reuse for next chunk)]
[if isLast: mark COMPLETE; getXferStatus returns success]
                                                       [if isLast: genNotif(notif_msg) → user]
-- end per chunk --
```

### READ

READ_REQ carries: `transfer_id`, initiator's pre-allocated slot index, `mode`, `[algo]`,
`slot_size`, `notif_msg`, and destination descriptors. `notif_msg` is the user-specified
notification string from `opt.notif` — delivered to the provider's user after the last RFREE
arrives (confirming all data has been received by the initiator). The provider fills each slot via NIXL_WRITE
with a chunk notification `_NIXLS_RCHUNK_{transfer_id, N, isLast}` attached. As each chunk
lands, the initiator drains it and immediately sends RFREE, carrying `isLast`, allowing the
provider to reuse the slot for the next chunk (backpressure and double buffering). On the last
chunk the initiator marks COMPLETE — at which point `getXferStatus` returns success. When the
last RFREE arrives at the provider, the provider frees the slot and delivers `notif_msg` to its
own user.

```
Initiator                                              Remote (provider)
---------                                              ------
[alloc local slot I]
genNotif(READ_REQ{transfer_id,              -->
                  slot_index, mode, [algo],
                  slot_size, notif_msg, dst_descs})

-- per chunk N --
                                                       [callback: alloc slot R, start fill]
                                                       fillSlot(dst_descs → slot_R) [async GPU]
                                                       checkFill(slot_R_addr) == DONE
                                                       postXferReq(NIXL_WRITE → slot_I,
                                   <--                 notif=_NIXLS_RCHUNK_{transfer_id, N, isLast})

[callback: chunk notif arrived]
drainSlot(slot_I → final_dst_N)    [async GPU]
checkDrain(slot_I_addr) == DONE
genNotif(_NIXLS_RFREE_{transfer_id, slot_I, isLast})  -->
[if isLast: mark COMPLETE; getXferStatus returns success]
                                                       [callback: free slot_R (reuse for next chunk)]
                                                       [if isLast: genNotif(notif_msg) → user]
-- end per chunk --
```

---

### Marshal Backends API

`nixlMarshalBackend` defines what each implementation must provide:

- **Capabilities:** `getSupportedMems()` — memory types accepted for staging;
`queryRequirements(MarshalRequirements&)` — pure virtual; each backend reports its ideal slot
size; `staging_backend` returns no entries (whole buffer = one slot); `marshalQuery` aggregates
across all compiled-in backends by instantiating each and calling this method
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

A static `prepare()` helper registers a single callback for the `_NIXLS_` prefix before the
base constructor runs, capturing a weak pointer to `nixlServiceAgentData`:

```cpp
nixlServiceAgent::nixlServiceAgent(const std::string& name, nixlServiceAgentConfig cfg)
    : nixlServiceAgent(name, prepare(std::move(cfg)))
```

The base agent routes matching notifications to the registered callback at arrival; all others
reach the user's `getNotifs()` unchanged. No backend registers its own callback —
nixlServiceAgent owns all `_NIXLS_*` routing and dispatches internally by prefix.

### driveService and Per-Service Queues

Callbacks must not call NIXL agent methods (`genNotif`, `createXferReq`, `postXferReq`, etc.) —
those carry locks and network state that are unsafe to invoke from the UCX progress thread.
Callbacks may update plain data structures under a mutex (slot pool bookkeeping, atomic handle
state), but all networking operations are queued as `ServiceWorkItem`s for execution on a safe
thread. Each sub-service (marshal, future encrypt, …) owns its own queue; the callback
dispatches by prefix into the appropriate queue, decoupling sub-services from each other.

Two progress models are supported, mirroring nixlAgent's own progress-thread duality:

`driveService()` is a private method — the canonical internal driver. It drains the service
work queue and collects user-visible notifications. Both `getXferStatus()` and `getNotifs()`
call it before doing their respective reads, so the standard nixlAgent polling idiom works
without any extra calls.

**No-progress-thread mode:** `getXferStatus()` and `getNotifs()` both call `driveService()`
on the caller's thread. No background thread needed for correctness.

**Progress-thread mode (Phase 4):** a background thread per sub-service runs `driveService()`
in a loop. `getXferStatus()` and `getNotifs()` become pure queries; `driveService()` inside
each is a no-op because the thread keeps the queue empty.

### State Machine and getXferStatus

`getXferStatus(nixlServiceXferReqH*)` calls `driveService()` then reads the completion flag set
when the queue processes the final protocol step. In progress-thread mode it becomes a pure
status read.

---

## File Structure

### New: `src/services/`


| File                               | Purpose                                                                                                                |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `nixl_service_types.h`             | `nixl_service_mode_t`, `nixl_comp_algo_t`, handle types, config, opt args, `MarshalRequirements`                       |
| `nixl_service.h`                   | `nixlServiceAgent` class declaration; `nixlService::marshalQuery`                                                      |
| `nixl_service_data.h`              | Pimpl `nixlServiceAgentData`: slot pools, transfer maps, work queue, marshal backend                                   |
| `nixl_service.cpp`                 | `nixlServiceAgent` implementation: mem registration, metadata exchange, transfer lifecycle                              |
| `marshal/marshal_backend.h`        | `nixlMarshalBackend` abstract interface                                                                                |
| `marshal/marshal_protocol.h/cpp`   | `nixlMarshalProtocol`: STAGE_BOTH wire protocol (dispatch, post, driveQueue, handlers)                                 |
| `marshal/marshal_utils.h`          | Inline pack/unpack helpers and `_NIXLS_*` wire message prefixes                                                        |
| `marshal/staging/staging_backend.h/cpp` | STAGE_BOTH: fill/drain via `cudaMemcpyAsync` gather/scatter                                                      |
| `marshal/nvcomp/nvcomp_backend.h/cpp`   | COMPRESS: fill/drain via nvCOMP ANS compress/decompress *(Phase 3)*                                              |
| `meson.build`                      | Conditional on `build_nixl_service`; add `nvcomp_path` for Phase 3                                                    |


### Modified Files


| File                 | Change                                                                    |
| -------------------- | ------------------------------------------------------------------------- |
| `src/api/cpp/nixl.h` | Add `virtual ~nixlAgent()`                                                |
| `meson_options.txt`  | Add `nvcomp_path` (string), `build_nixl_service` (boolean, default false) |
| `src/meson.build`    | Add conditional `subdir('services')`                                      |


---

## Implementation Phases

### Phase 0: Scaffolding ✅

Add `virtual ~nixlAgent()`, build options, and `src/services/` directory stub.

### Phase 1: Types and Headers ✅

Define `nixl_service_types.h`, `marshal_backend.h`, `nixl_service.h`, `nixl_service_data.h`.
Wire message structs (RTS, CTS, READ_REQ, slot-freed, final notif) with their fields;
`nixlMarshalBackend` interface including `fillSlot`/`drainSlot`/`checkFill`/`checkDrain`.

### Phase 2: staging_backend + STAGE_BOTH ✅

Implement `staging_backend` via `cudaMemcpyAsync`. Implement DIRECT passthrough and full
STAGE_BOTH WRITE and READ state machines: RTS/CTS handshake, NIXL_WRITE with intermediate
notification, drain, SLOT_FREED, final notification. Callbacks are minimal — they enqueue work
items into a per-service queue and return; `driveService()` drains the queue on the user's
thread. GPU op polling is synchronous in the queue drain (not in the callback). Single staging
slot per transfer (data fits in one slot).

**Milestone:** single-process two-agent WRITE and READ with non-contiguous descriptors
verified; nixlAgent tests pass in DIRECT mode. ✅

### Phase 3: nvcomp_backend + COMPRESS

Implement `nvcomp_backend` via nvCOMP ANS. Add an `algo` field to RTS and READ_REQ wire
messages so the receiver selects the correct backend. Wire protocol otherwise unchanged from
Phase 2; only `fillSlot`/`drainSlot` and their async completion differ. `nvcomp_backend`
populates `algSlotSizes` for ANS in `queryRequirements`.

Two implementation constraints to observe:
- **Workspace isolation:** nvCOMP requires a temporary workspace per active transfer
  (`getCompressTempSize`). This workspace must never be shared across concurrent transfers.
  two concurrent transfers sharing a workspace will silently corrupt each other's intermediate
  state. Each `nixlServiceXferReqH` owns its workspace for the lifetime of the transfer.
- **GPUNETIO stream ordering:** `nvcomp_backend` uses a compression CUDA stream and a transfer
  stream ordered via `cudaEventRecord`/`cudaStreamWaitEvent`. This ordering must be validated
  against the GPUNETIO backend, which has its own internal stream model that may
  interact with external stream dependencies.

**Milestone:** WRITE and READ compress/decompress verified at multiple transfer sizes.

### Phase 4: Multi-Chunk Pipelining + Per-Service Thread

**Pipelining:** lift the single-slot limitation. Each chunk is sent as a separate NIXL_WRITE
with `isLast=0`; the final chunk carries `isLast=1`. The `isLast` flag is already in the wire
(WCHUNK/RCHUNK and WFREE/RFREE); the main work is allocating and tracking multiple slots
concurrently and adding chunk counters to `InboundTransferState`.

**Per-service thread:** add a background thread per sub-service that drains its queue
autonomously, keeping GPU polling off the user's thread.

**Milestone:** transfers larger than one slot complete correctly with multiple chunks in-flight.

### Phase 5: Python Bindings

Bind `nixlServiceAgent`, config, opt args, handle type, and `marshalQuery`.
**Milestone:** DIRECT, STAGE_BOTH, and COMPRESS end-to-end in Python.

---

## Out of Scope

- STAGE_INIT, STAGE_TARGET, ENC, ENC_COMP — future `nixlMarshalBackend` implementations
- Multiple compression algorithms beyond ANS
- CPU compression backend (DRAM staging via `getSupportedMems`)

