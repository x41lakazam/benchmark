"""Host concurrency must preserve per-axis order and drain work on failure."""

import importlib.util
from pathlib import Path
import sys
import threading
import types
from unittest.mock import patch

import pytest


@pytest.fixture
def execution_cls():
    name = "_host_test.execution"
    gate = types.ModuleType("_host_test.utils.start_gate")
    gate.CUDAStartGate = object
    path = Path(__file__).parents[1] / "maestro/core/execution.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": types.ModuleType("torch"), gate.__name__: gate}):
        spec.loader.exec_module(module)
    return module.HostExecution


def test_axes_overlap_but_each_axis_stays_serial(execution_cls):
    a, b = (types.SimpleNamespace(name=name) for name in ("a", "b"))
    rendezvous = threading.Barrier(2)
    done = threading.Event()
    calls = []

    def first():
        rendezvous.wait(timeout=3)  # Fails if different axes execute serially.
        calls.append("a1")
        done.set()

    def second():
        assert done.is_set()
        calls.append("a2")

    blocks = [
        types.SimpleNamespace(axis=a, run=first),
        types.SimpleNamespace(axis=a, run=second),
        types.SimpleNamespace(axis=b, run=lambda: rendezvous.wait(timeout=3)),
    ]
    executor = execution_cls([a, b])
    try:
        executor.execute(
            blocks,
            2,
            types.SimpleNamespace(barrier=lambda: None),
            types.SimpleNamespace(run_block=lambda block: block.run()),
        )
        assert calls == ["a1", "a2", "a1", "a2"]
    finally:
        executor.close()


def test_worker_failure_drains_other_axes_before_return(execution_cls):
    rendezvous = threading.Barrier(2)
    completed = threading.Event()

    def fail():
        rendezvous.wait(timeout=3)
        raise ValueError("worker failed")

    def finish():
        rendezvous.wait(timeout=3)
        completed.set()

    blocks = [
        types.SimpleNamespace(axis=types.SimpleNamespace(name=name), run=fn)
        for name, fn in (("a", fail), ("b", finish))
    ]
    executor = execution_cls(blocks)
    try:
        with pytest.raises(ValueError, match="worker failed"):
            executor.execute(
                blocks,
                1,
                types.SimpleNamespace(barrier=lambda: None),
                types.SimpleNamespace(run_block=lambda block: block.run()),
            )
        assert completed.is_set()
    finally:
        executor.close()
