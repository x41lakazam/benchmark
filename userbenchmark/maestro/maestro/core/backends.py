"""Registered backend components and validation, without runtime construction."""

from dataclasses import dataclass
from functools import partial

from .allocator import TorchAllocator, NCCLAllocator
from .axis import TorchAxis, HostAxis
from .execution import CUDAExecution, HostExecution
from .profilers import CuptiProfiler
from .profilers.host import HostProfiler
from .utils.distributed import _TorchDistUtils, _TorchNCCLDistUtils


@dataclass(frozen=True)
class BackendSpec:
    axis_cls: type
    dist_factory: object
    execution_cls: type
    allocator_cls: type
    device: str
    default_profiler: str
    dtypes: tuple = ("float16", "float32")
    axis_params: tuple = ()
    supports_registration: bool = False
    supports_axis_envs: bool = False


BACKENDS = {}
PROFILERS = {"cupti": CuptiProfiler, "host": HostProfiler}


def register_backend(name, backend, spec):
    key = (name, backend)
    if key in BACKENDS:
        raise ValueError(f"Backend already registered: {key}")
    BACKENDS[key] = spec


def register_profiler(name, profiler_cls):
    if name in PROFILERS:
        raise ValueError(f"Profiler already registered: {name}")
    PROFILERS[name] = profiler_cls


def get_backend(axis_config):
    key = (axis_config["name"], axis_config["backend"])
    try:
        return BACKENDS[key]
    except KeyError:
        raise ValueError(f"Unknown axis backend {key}; registered: {list(BACKENDS)}") from None


def validate_backend(config):
    from .block import BlockRegistry

    spec = get_backend(config["backends"]["axis"])
    if config["dtype"] not in spec.dtypes:
        raise ValueError(f"Backend does not support dtype {config['dtype']}; choose {spec.dtypes}")
    if config["register_buffers"] and not spec.supports_registration:
        raise ValueError("Selected backend does not support register_buffers")
    if any(config["axis_envs"].values()) and not spec.supports_axis_envs:
        raise ValueError("Selected backend does not support per-axis environment overrides")
    for params in config["axis_params"].values():
        if set(params) - set(spec.axis_params):
            raise ValueError(f"Unsupported process-group options for this backend: {params}")
    profiler_name = config["backends"]["profiler"]["name"]
    if profiler_name not in PROFILERS:
        raise ValueError(f"Unknown profiler: {profiler_name}")
    if PROFILERS[profiler_name].execution_mode != spec.execution_cls.mode:
        raise ValueError(f"Profiler {profiler_name} cannot measure {spec.execution_cls.mode} execution")
    for pattern in config["patterns"]:
        for op in pattern["ops"]:
            BlockRegistry.get_block_cls(op["block"]).validate_backend(spec)
    return spec


register_backend(
    "torch",
    "nccl",
    BackendSpec(
        TorchAxis,
        _TorchNCCLDistUtils,
        CUDAExecution,
        NCCLAllocator,
        "cuda",
        "cupti",
        dtypes=("float16", "float32", "bfloat16"),
        axis_params=("traffic_class",),
        supports_registration=True,
        supports_axis_envs=True,
    ),
)
register_backend(
    "torch",
    "gloo",
    BackendSpec(
        HostAxis,
        partial(_TorchDistUtils, backend="gloo"),
        HostExecution,
        TorchAllocator,
        "cpu",
        "host",
    ),
)
