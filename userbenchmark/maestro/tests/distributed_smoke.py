"""Run with torchrun --nproc-per-node=4 -m userbenchmark.maestro.tests.distributed_smoke."""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from ..maestro.core.backends import get_backend, validate_backend
from ..maestro.core.block import BlockRegistry
from ..maestro.core.config_parser import parse_config
from ..maestro.core.utils.distributed import nccl_torch_dist_utils
from ..maestro.main import _run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = Path(__file__).parents[1] / "examples/config_files/gloo.yaml"
    raw = yaml.safe_load(config_path.read_text())
    try:
        # Assert capability failures before creating a runtime's process groups.
        for change in (
            {"register_buffers": True},
            {"dtype": "bfloat16"},
            {"backends": {"axis": {"backend": "gloo"}, "profiler": {"name": "cupti"}}},
            {
                "patterns": [
                    {"ops": [{"block": "torch_gemm", "axis": "rows", "mat_a_shape": [2, 2], "mat_b_shape": [2, 2]}]}
                ]
            },
        ):
            config = parse_config({**deepcopy(raw), **change})
            try:
                validate_backend(config)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Unsupported configuration accepted: {change}")

        # Existing ring implementation must complete on the CPU without a CUDA
        # fence; rank-specific payloads detect missing waits and wrong peers.
        spec = get_backend({"name": "torch", "backend": "gloo"})
        first, second = spec.dist_factory(), spec.dist_factory()
        try:
            allocator = spec.allocator_cls("cpu", torch.float32)
            axis = spec.axis_cls(
                [list(range(dist.get_world_size()))], name="ring", dist_utils=first, allocator=allocator
            )
            ring = BlockRegistry.get_block_cls("torch_send_recv_ring").from_api_params(
                axis, "ring", {"vector_size": 64}
            )
            assert ring.src_buf.device.type == "cpu" and ring.src_buf.element_size() == 4
            ring.src_buf.fill_(dist.get_rank())
            ring.run()
            assert torch.all(ring.dst_buf == (dist.get_rank() - 1) % dist.get_world_size())
            first.destroy()
            second.barrier()  # Destroying one runtime must leave another usable.
            assert dist.get_backend() == "nccl"
        finally:
            first.destroy()
            second.destroy()

        # Use the production runner repeatedly in one process. Group ownership
        # must not tear down bootstrap state needed by the next invocation.
        for backend, dtype, registered in (
            ("gloo", "float32", False),
            ("nccl", "float16", False),
            ("gloo", "float16", False),
            ("nccl", "float16", True),
        ):
            config = deepcopy(raw)
            config["backends"]["axis"]["backend"] = backend
            config["dtype"] = dtype
            config["register_buffers"] = registered
            if backend == "nccl":
                config["patterns"].append(
                    {
                        "name": "compute_overlap",
                        "ops": [
                            {"block": "torch_all_reduce", "axis": "rows", "name": "reduce", "vector_size": "64KiB"},
                            {
                                "block": "torch_gemm",
                                "axis": "columns",
                                "name": "gemm",
                                "mat_a_shape": [128, 128],
                                "mat_b_shape": [128, 128],
                            },
                        ],
                    }
                )
            output = args.output_dir / f"{backend}-{dtype}-registered-{registered}.json"
            config["save_results_to"] = str(output)
            result = _run_benchmark(config)
            assert len(result) == (10 if backend == "nccl" else 8)
            assert set(result.backend) == {backend}
            assert set(result.dtype) == {dtype}
            communication = result[result.block_name != "gemm"]
            assert set(communication.block_size) == {"64K"}
            domain = "host_blocking_call" if backend == "gloo" else "cuda_activity"
            assert set(result.timing_domain) == {domain}
            assert (result.min_latency_ns > 0).all()
            assert (result.min_latency_ns <= result.avg_latency_ns).all()
            assert (result.avg_latency_ns <= result.max_latency_ns).all()
            assert communication.avg_bandwidth_gbps.notna().all()
            serial = result[result.pattern == "serial"]
            assert serial.avg_overlap_pct.isna().all()
            assert serial.avg_shared_bw_gbps.isna().all()
            parallel = result[result.pattern != "serial"]
            assert parallel.avg_overlap_pct.between(0, 100).all()
            assert dist.is_initialized()
        if dist.get_rank() == 0:
            print(f"Distributed smoke passed: {torch.__version__}, {torch.cuda.get_device_name()}")
    finally:
        nccl_torch_dist_utils.destroy()


if __name__ == "__main__":
    main()
