# Backend integration after the Gloo experiment

This branch extends the fifth Maestro PR (`6b98aec3`). Gloo now runs through the
normal Maestro CLI and TorchBench entry point, using the existing communication
blocks, orchestrator, and result aggregation.

| Original obstacle | Integration change |
| --- | --- |
| `axis.backend` ignored | `BackendSpec` is selected by both axis name and backend; unknown pairs fail explicitly. |
| Import already initializes NCCL | Eager NCCL remains the bootstrap policy; runtime utilities create owned subgroups without initializing the default group again. Configuration timestamp coordination can receive a utility explicitly. |
| CUDA/fp16 buffers hardcoded | An injected allocator controls placement and dtype, including block fallback allocations; parsing uses the configured dtype's element size. NCCL registration moved into its allocator. |
| Orchestrator directly schedules CUDA | `CUDAExecution` retains the gate/fences; `HostExecution` runs serial blocks per axis on concurrent host workers, draining them between iterations. |
| Completion assumes a GPU stream | `Axis.complete_work` orders CUDA streams or waits on a host worker; the same ring block works with both. |
| CUPTI-only profiling | A registered host profiler records complete blocking calls; both profilers use the same discovery and reporting code. Results carry a timing domain. |
| Capabilities scattered across concrete classes/names | Backend settings and block device requirements are validated before runtime creation; even-split requirements are inherited block metadata. Torch modules no longer require a concrete CUDA axis type. |
| Global group ownership | Axis utilities are injected per instance, runtime cleanup destroys only owned subgroups, and CLI shutdown separately closes bootstrap. |
| Extension docs use outdated hooks | Developer examples describe the actual axis constructor, profiler hooks, allocator, execution, and registration interfaces. |

The implementation is intentionally scoped to one backend/device/dtype per run.
It retains Torch tensors and GPU-equipped startup; it does not implement mixed
backends per axis, arbitrary buffer representations, or GPU-tensor Gloo.
New backend factories can reuse the registered components or supply their own.

Gloo supports the six existing communication blocks with CPU float16/float32
buffers. CUDA-only GEMM, registered buffers, traffic-class options, and per-axis
environment overrides are rejected for Gloo. Existing NCCL defaults, eager
initialization, gate ordering, nonblocking P2P stream fences, and the weighted
shared-bandwidth formula are retained.

Host timing includes dispatch, waiting, and host scheduling. Mean host overlap
describes call intervals; normalized shared bandwidth does not measure physical
bus traffic. It is a different timing domain from CUPTI activities.

See [backend configuration](userbenchmark/maestro/docs/doc.md#gloo-and-backend-selection)
and [extension interfaces](userbenchmark/maestro/docs/developer.md#adding-a-new-axis-backend).
The distributed smoke check runs Gloo and NCCL repeatedly in one process, checks
independent group lifetimes and ring payloads, and exercises both normal and
registered NCCL buffers:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=4 \
  -m userbenchmark.maestro.tests.distributed_smoke --output-dir /tmp/maestro-backends
```

Validated on Gaia on 2026-10-15 (job 70854): four ranks on one H100 node,
`nvcr.io/nvidia/pytorch:25.09-py3`, PyTorch `2.9.0a0+50eac811a6.nv25.09`.
All 48 tests passed, including the CUDA start-gate test. The distributed smoke
passed for Gloo float16/float32 and NCCL with/without registered buffers, including
communication overlapping GEMM. Both CLI entry points completed successfully;
result files contained finite timings and the TorchBench output had 52 finite
metrics. These are execution and result-consistency checks, not performance claims
or multi-node validation.
