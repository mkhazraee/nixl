#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import pickle
import random

import numpy as np
import torch

from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument("--backend", type=str, default="UCX")
    parser.add_argument(
        "--mode",
        type=str,
        default="initiator",
        help="Local IP in target, peer IP (target's) in initiator",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # initiator use default port
    listen_port = args.port
    if args.mode != "target":
        listen_port = 0

    if args.use_cuda:
        torch.set_default_device("cuda:0")
    else:  # To be sure this is the default
        torch.set_default_device("cpu")

    config = nixl_agent_config(True, True, listen_port, backends=[args.backend])

    # Allocate memory and register with NIXL
    try:
        agent = nixl_agent(args.mode, config)
    except Exception as e:
        logger.exception("Failed to create NIXL agent: %s", e)
        exit(1)

    # Use a single 2D tensor with 16 tensors of size 32
    if args.mode == "target":
        tensor = torch.ones((16, 32), dtype=torch.float32)
    else:
        tensor = torch.zeros((16, 32), dtype=torch.float32)

    logger.info(
        "Running test with tensor shape %s in mode %s", tuple(tensor.shape), args.mode
    )

    # Register the single 2D tensor. Transfers can be issued from any location within the registered memory.
    # The fewer, larger registrations, the better—this reduces kernel calls and internal lookups.
    try:
        reg_descs = agent.register_memory(tensor)
        if not reg_descs:
            logger.error("Memory registration failed.")
            exit(1)
    except Exception as e:
        logger.exception("Memory registration failed: %s", e)
        exit(1)

    # Target code: its memory is read first, then written at randomly selected locations, then read again.
    if args.mode == "target":
        ready = False

        # Build transfer descriptors by unraveling the first dimension into a list of tensors
        try:
            target_tensors = [tensor[i, :] for i in range(tensor.shape[0])]
            target_descs = agent.get_xfer_descs(target_tensors)
            if not target_descs:
                logger.error("Failed to build target transfer descriptors.")
                exit(1)
            target_desc_str = agent.get_serialized_descs(target_descs)
        except Exception as e:
            logger.exception("Preparing target descriptors failed: %s", e)
            exit(1)

        # Alternatively, pass minimal layout information so the initiator can generate descriptors locally.
        base_addr = tensor.data_ptr()
        tensors = int(tensor.shape[0])  # 16
        tensor_size = int(tensor.shape[1] * tensor.element_size())  # bytes per tensor
        dev_id = tensor.get_device()
        if dev_id == -1:
            dev_id = 0
        mem_str = "cuda" if str(tensor.device).startswith("cuda") else "cpu"

        # Send descriptor list + layout to the initiator after its metadata is received.
        try:
            while not ready:
                ready = agent.check_remote_metadata("initiator")
            agent.send_notif(
                "initiator",
                pickle.dumps(
                    (
                        target_desc_str,
                        (base_addr, tensors, tensor_size, dev_id, mem_str),
                    )
                ),
            )
        except Exception as e:
            logger.exception("Send of descriptors to initiator failed: %s", e)
            exit(1)

        # Wait for transfer notifications by polling; exact match for READs, startswith("Write") for WRITEs
        expected_reads = {
            b"Read idx 0,4,8",
            b"Read idx 1,5,9",
            b"Read idx 2,6,10",
            b"Read idx 3,7,11",
            b"Read idx 0,4,8 again",
            b"Read idx 1,5,9 again",
            b"Read idx 2,6,10 again",
            b"Read idx 3,7,11 again",
        }
        remaining_writes = 2
        try:
            logger.info("Waiting for transfers (4 READs and 2 WRITEs)")
            while expected_reads or remaining_writes > 0:
                notif_map = agent.get_new_notifs()
                if "initiator" in notif_map:
                    for msg in notif_map["initiator"]:
                        if msg in expected_reads:
                            expected_reads.remove(msg)
                        elif msg.startswith(b"Write") and remaining_writes > 0:
                            remaining_writes -= 1
        except Exception as e:
            logger.exception("Polling notifications failed: %s", e)
            exit(1)

        # Verify target tensor contents: last 4 tensors should have at least some zeros
        tail = tensor[12:, :]
        if not torch.any(tail == 0.0):
            logger.error("Target data verification failed: no zeros detected.")
            exit(1)
        logger.info("Target data verification passed (zeros found in last 4 tensors)")

    # Initiator code: reads target memory and then writes to randomly selected locations, then reads again.
    else:
        logger.info("Initiator sending to %s", args.ip)
        try:
            # Exchange metadata after registrations, because they carry relevant information.
            # Since the target process starts first and our registration is done, this is proper.
            agent.send_local_metadata(args.ip, args.port)
            agent.fetch_remote_metadata("target", args.ip, args.port)
        except Exception as e:
            logger.exception("Metadata exchange (fetch/send) failed: %s", e)
            exit(1)

        try:
            notifs = agent.get_new_notifs()
            while len(notifs) == 0:
                notifs = agent.get_new_notifs()
            target_descs_ser, layout_info = pickle.loads(notifs["target"][0])
            target_descs = agent.deserialize_descs(target_descs_ser)
        except Exception as e:
            logger.exception("Receiving target descriptors failed: %s", e)
            exit(1)

        # Build local transfer descriptors by unraveling the first dimension into a list of tensors
        try:
            initiator_tensors = [tensor[i, :] for i in range(tensor.shape[0])]
            initiator_descs = agent.get_xfer_descs(initiator_tensors)
            if not initiator_descs:
                logger.exception("Initiator's local descriptors creation failed.")
                exit(1)
        except Exception as e:
            logger.exception("Initiator's local descriptors creation failed: %s", e)
            exit(1)

        # Ensure remote metadata has arrived from fetch
        ready = False
        while not ready:
            try:
                ready = agent.check_remote_metadata("target")
            except Exception as e:
                logger.exception("Checking of target metadata failed: %s", e)
                exit(1)

        logger.info("Ready for transfers")

        # 1) Parallel transfers using prep_xfer + make_prepped_xfer when blocks are known in advance.
        # As an example, make 4 transfers of 3 tensors each, spaced 4 apart, using reversed ordering for remote blocks.
        try:
            local_side = agent.prep_xfer_dlist("", initiator_descs)
            remote_side = agent.prep_xfer_dlist("target", target_descs)
        except Exception as e:
            logger.exception("Preparation of xfer dlists failed: %s", e)
            exit(1)

        handles = []
        read_handles = []
        read_handles_2 = []
        try:
            for start in range(4):
                idxs = [start, start + 4, start + 8]
                notif = f"Read idx {start},{start + 4},{start + 8}".encode()
                read_handles.append(
                    agent.make_prepped_xfer(
                        "READ", local_side, idxs, remote_side, idxs[::-1], notif
                    )
                )
            read_handles_2 = list(read_handles)
            handles.extend(read_handles)
        except Exception as e:
            logger.exception("Creating READ handles failed: %s", e)
            exit(1)

        # 2) Parallel transfers using initialize_xfer when locations are chosen at transfer time
        #    (or when there is no notion of fixed blocks). As an example, randomly select which half
        #    of each tensor to write, using 2 descriptors in each of the two transfers.
        write_handles = []
        # Build local/remote descriptors for both WRITE requests
        base_addr, tensors, tensor_size, remote_dev, remote_mem = layout_info
        local_mem = "cuda" if str(tensor.device).startswith("cuda") else "cpu"
        local_dev = tensor.get_device()
        if local_dev == -1:
            local_dev = 0

        random.seed(0)
        starts_bytes = {
            r: (0 if random.randint(0, 1) == 0 else tensor_size // 2)
            for r in range(12, 16)
        }
        half_len = int(tensor_size // 2)

        # First WRITE: tensors 12 and 13 (using Python list/tuple descriptors)
        r0, r1 = 12, 13
        off0, off1 = starts_bytes[r0], starts_bytes[r1]
        write_notif0 = f"Write tensors {r0}({'first' if off0 == 0 else 'second'}),{r1}({'first' if off1 == 0 else 'second'})".encode()
        local_w0 = [
            (tensor[r0, :].data_ptr() + off0, half_len, local_dev),
            (tensor[r1, :].data_ptr() + off1, half_len, local_dev),
        ]
        remote_w0 = [
            (base_addr + r0 * tensor_size + off0, half_len, remote_dev),
            (base_addr + r1 * tensor_size + off1, half_len, remote_dev),
        ]
        local_w0_d = agent.get_xfer_descs(local_w0, mem_type=local_mem)
        remote_w0_d = agent.get_xfer_descs(remote_w0, mem_type=remote_mem)

        # Second WRITE: tensors 14 and 15 (using NumPy Nx3 descriptors for performance benefits over list/tuple)
        r2, r3 = 14, 15
        off2, off3 = starts_bytes[r2], starts_bytes[r3]
        write_notif1 = f"Write tensors {r2}({'first' if off2 == 0 else 'second'}),{r3}({'first' if off3 == 0 else 'second'})".encode()
        local_w1_np = np.array(
            [
                [tensor[r2, :].data_ptr() + off2, half_len, local_dev],
                [tensor[r3, :].data_ptr() + off3, half_len, local_dev],
            ],
            dtype=np.uint64,
        )
        remote_w1_np = np.array(
            [
                [base_addr + r2 * tensor_size + off2, half_len, remote_dev],
                [base_addr + r3 * tensor_size + off3, half_len, remote_dev],
            ],
            dtype=np.uint64,
        )
        local_w1_d = agent.get_xfer_descs(local_w1_np, mem_type=local_mem)
        remote_w1_d = agent.get_xfer_descs(remote_w1_np, mem_type=remote_mem)

        # Prepare both WRITE requests
        try:
            xfer_w0 = agent.initialize_xfer(
                "WRITE", local_w0_d, remote_w0_d, "target", write_notif0
            )
            xfer_w1 = agent.initialize_xfer(
                "WRITE", local_w1_d, remote_w1_d, "target", write_notif1
            )
            write_handles = [xfer_w0, xfer_w1]
            handles.extend(write_handles)
        except Exception as e:
            logger.exception("Preparing WRITE handles failed: %s", e)
            exit(1)

        # Post all READs in parallel and wait (no ordering guarantees across them).
        try:
            for h in read_handles:
                st = agent.transfer(h)
                if st == "ERR":
                    logger.error("Posting READ failed.")
                    exit(1)
            while read_handles:
                # iterate over a snapshot to safely remove completed handles
                for h in list(read_handles):
                    st = agent.check_xfer_state(h)
                    if st == "ERR":
                        logger.error("A READ transfer got to Error state.")
                        exit(1)
                    if st == "DONE":
                        read_handles.remove(h)
        except Exception as e:
            logger.exception("Posting/awaiting READs failed: %s", e)
            exit(1)

        # Applications can enforce ordering by waiting for some transfers to finish before starting others,
        # Now post both WRITEs in parallel (without ordering guarantees) and wait.
        try:
            for h in write_handles:
                st = agent.transfer(h)
                if st == "ERR":
                    logger.error("Posting WRITE failed.")
                    exit(1)
            while write_handles:
                for h in list(write_handles):
                    st = agent.check_xfer_state(h)
                    if st == "ERR":
                        logger.error("A WRITE transfer errored.")
                        exit(1)
                    if st == "DONE":
                        write_handles.remove(h)
        except Exception as e:
            logger.exception("Posting/awaiting WRITEs failed: %s", e)
            exit(1)

        # Repost all READs with new notifications (no re-preparation needed).
        # Example use case is when some data is getting updated, e.g., model parameters, and we want to read the updated data from the same locations to the same locations.
        try:
            for start in range(4):
                # read_handle_2 is a list, so the same order is maintained
                notif2 = f"Read idx {start},{start + 4},{start + 8} again".encode()
                st = agent.transfer(read_handles_2[start], notif2)
                if st == "ERR":
                    logger.error("Reposting READ failed.")
                    exit(1)
            while read_handles_2:
                for h in list(read_handles_2):
                    st = agent.check_xfer_state(h)
                    if st == "ERR":
                        logger.error("A reposted READ transfer errored.")
                        exit(1)
                    if st == "DONE":
                        read_handles_2.remove(h)
        except Exception as e:
            logger.exception("Reposting/awaiting READs (second round) failed: %s", e)
            exit(1)

        # Final verification on initiator: first 12 tensors should be ones
        check = torch.zeros_like(tensor)
        check[:12, :] = 1.0
        if not torch.allclose(tensor, check):
            logger.error("Initiator final data verification failed.")
            exit(1)
        logger.info("Initiator final data verification passed")

    # Tear down. The Python garbage collector will release transfer handles, but it's better to be explicit.
    # Metadata and registrations will also be released by the NIXL agent during destruction, but explicit cleanup is clearer.
    # (Metadata removal can also be done dynamically at runtime, for example to remove a failed node from possible destinations.)
    if args.mode != "target":
        try:
            agent.remove_remote_agent("target")
            for h in handles:
                agent.release_xfer_handle(h)
            agent.release_dlist_handle(local_side)
            agent.release_dlist_handle(remote_side)
            agent.invalidate_local_metadata(args.ip, args.port)
        except Exception as e:
            logger.exception("Tear down (metadata/transfer handles) failed: %s", e)

    try:
        agent.deregister_memory(reg_descs)
    except Exception as e:
        logger.exception("Deregisteration of memory failed: %s", e)

    logger.info("Test Complete.")
