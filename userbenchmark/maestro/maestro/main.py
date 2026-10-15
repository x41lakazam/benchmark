import click
from pathlib import Path
import yaml
import pprint
from .core.config_parser import parse_config
from .core.orchestrator import Orchestrator
import torch
from .core.backends import validate_backend, PROFILERS
from .core.utils.distributed import nccl_torch_dist_utils
from .core.utils.logging import get_logger, get_root_rank_logger


logger = get_logger(__name__)
root_logger = get_root_rank_logger()

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli():
    """
    Maestro CLI - A benchmarking framework for parallel communication/compute operations.

    Maestro enables accurate performance measurement of AI workloads by benchmarking
    parallel operations on the same GPU, rather than isolated micro-benchmarks.

    Use 'python -m maestro <command> --help' for more information on a specific command.
    """
    return

@cli.command(short_help="List all available blocks.")
def list_blocks():
    """
    List all available blocks registered in the BlockRegistry.

    Blocks are the fundamental building units in Maestro patterns. Each block
    represents a discrete operation (e.g., NCCL collective, GEMM, custom kernel)
    that can be composed into workload patterns.

    Example:
        $ python -m maestro list-blocks
    """
    from .core.block import BlockRegistry
    print("Available blocks:")
    for block in BlockRegistry.get_all_blocks():
        print(f"- {block.registry_name}")

@cli.command(short_help="List all available operation presets.")
def list_op_presets():
    """
    List all available operation presets registered in the OpPresetRegistry.

    Op presets define pre-configured operation parameters that can be reused
    across different patterns. They simplify configuration by providing
    commonly-used operation settings.

    Example:
        $ python -m maestro list-op-presets
    """
    from .core.op_preset import OpPresetRegistry
    print("Available op presets:")
    for op_preset in OpPresetRegistry.get_all_op_presets():
        print(f"- {op_preset.name}")

@cli.command(short_help="Run benchmarks with a config file.")
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='Path to pattern/config file.')
@click.option('--save-results-to', type=str, default=None, help='Output file path (csv, json). Overrides save_results_to in the config file.')
def run(config: Path, save_results_to: str):
    """
    Run benchmarks for distributed patterns defined in a configuration file.

    This command parses the provided YAML configuration file, initializes the
    specified backends (axis and profiler), and executes all defined patterns
    through the Orchestrator. Results are collected and optionally saved to
    a file.

    The configuration file should define:

    \b
    - patterns: List of workload patterns to benchmark
    - axes: Axis configuration for distributed execution
    - backends: Backend settings (axis type, profiler)
    - warmup_iters: Number of warmup iterations before measurement
    - iters: Number of measurement iterations
    - save_results_to: Optional output file path (csv, json)

    Supported output formats: .csv, .json

    The --save-results-to CLI option, if provided, takes precedence over the
    save_results_to value in the config file.

    Example:
        $ python -m maestro run -c configs/allreduce_gemm.yaml
        $ python -m maestro run -c configs/allreduce_gemm.yaml --save-results-to /tmp/results_%D.csv
    """
    return _run_benchmark(config, save_results_to)


def _run_benchmark(config: Path, save_results_to: str = None):
    config = parse_config(config, save_results_to_override=save_results_to)

    spec = validate_backend(config)
    profiler_config = config["backends"]["profiler"]
    profiler = PROFILERS[profiler_config["name"]](output_dir=profiler_config["output_dir"])
    allocator = spec.allocator_cls(spec.device, getattr(torch, config["dtype"]), config["register_buffers"])
    dist_utils = spec.dist_factory()
    orchestrator = None
    try:
        root_logger.info(f"Running with config:\n{pprint.pformat(config)}")
        orchestrator = Orchestrator(
            patterns=config["patterns"], axes_cfg=config["axes"],
            axis_envs=config["axis_envs"], axis_params=config["axis_params"],
            profiler=profiler, axis_cls=spec.axis_cls, dist_utils=dist_utils,
            allocator=allocator, execution_cls=spec.execution_cls,
            warmup_iters=config["warmup_iters"], iters=config["iters"],
            latency_precision=config["latency_precision"],
            register_buffers=config["register_buffers"], stop_on_error=config["stop_on_error"],
        )
        results_df = orchestrator.run_all_patterns()
        results_df["backend"] = config["backends"]["axis"]["backend"]
        results_df["timing_domain"] = profiler.timing_domain
        results_df["dtype"] = config["dtype"]
        output = config["save_results_to"]
        if output and dist_utils.get_rank() == 0:
            if output.suffix.lower() == ".csv":
                results_df.to_csv(output, index=False)
            else:
                results_df.to_json(output, orient="records", index=False)
            root_logger.info(f"Saved results to {output.absolute()}")
        return results_df
    finally:
        if orchestrator is not None:
            orchestrator.destroy()
        else:
            allocator.close()
            dist_utils.destroy()


if __name__ == '__main__':
    try:
        cli()
    finally:
        nccl_torch_dist_utils.destroy()
