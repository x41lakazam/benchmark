"""Execution policies own concurrency, iteration fences, and completion."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading

import torch

from .utils.start_gate import CUDAStartGate


class CUDAExecution:
    mode = "cuda"

    def __init__(self, axes):
        self._start_gate = CUDAStartGate()

    def execute(self, blocks, iters, dist_utils, profiler):
        if iters == 0:
            return
        dist_utils.barrier()
        axes = set(block.axis for block in blocks)
        with self._start_gate.hold(axes):
            for _ in range(iters):
                for block in blocks:
                    profiler.run_block(block)
                for axis in axes:
                    torch.cuda.current_stream().wait_stream(axis.stream)
                barrier_handle = dist_utils.barrier(async_op=True)
                for axis in axes:
                    with axis.use_axis():
                        barrier_handle.block_current_stream()
        self.synchronize()

    @staticmethod
    def synchronize():
        torch.cuda.synchronize()

    def close(self):
        self._start_gate.close()


class HostExecution:
    mode = "host"

    def __init__(self, axes):
        self._workers = ThreadPoolExecutor(max_workers=len(axes))

    def execute(self, blocks, iters, dist_utils, profiler):
        by_axis = defaultdict(list)
        for block in blocks:
            by_axis[block.axis.name].append(block)

        def run_axis(axis_blocks, start):
            start.wait(timeout=30)
            for block in axis_blocks:
                profiler.run_block(block)

        for _ in range(iters):
            dist_utils.barrier()
            start = threading.Barrier(len(by_axis) + 1)
            futures = [self._workers.submit(run_axis, axis_blocks, start) for axis_blocks in by_axis.values()]
            start.wait(timeout=30)
            # Drain every worker, including when one fails, before propagating
            # errors or destroying groups/buffers used by other axes.
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]

    @staticmethod
    def synchronize():
        # Each block finishes on its calling thread; execute joins all workers.
        pass

    def close(self):
        self._workers.shutdown(wait=True)
