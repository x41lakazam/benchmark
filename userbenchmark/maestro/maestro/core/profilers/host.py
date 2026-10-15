"""Measure blocking host calls; these spans include dispatch and waiting."""

import json
import threading
import time

from .base import profiler_ctx


class HostProfiler(profiler_ctx):
    execution_mode = "host"
    timing_domain = "host_blocking_call"

    def __init__(self, output_dir=None):
        super().__init__(output_dir)
        self._active = False
        self._lock = threading.Lock()

    def _enter(self):
        self._active = True

    def _exit(self, nosave=False):
        self._active = False

    def run_block(self, block):
        start = time.perf_counter_ns()
        block.run()
        end = time.perf_counter_ns()
        if self._active:
            with self._lock:
                self._save_results(
                    [
                        {
                            "kind": "host_call",
                            "name": block.name,
                            # Logical serial queue, not a hardware CUDA stream.
                            "stream_id": block.axis.name,
                            "start_ns": start,
                            "end_ns": end,
                            "duration_ns": end - start,
                        }
                    ]
                )

    def _save_trace(self, output_file):
        output_file.write_text(json.dumps(self.get_results(), indent=2))
