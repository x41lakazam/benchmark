from abc import ABC, abstractmethod
from typing import Iterator, ContextManager
from contextlib import contextmanager, nullcontext
import contextvars
import os
import torch

from .utils.logging import get_logger


logger = get_logger(__name__)

# Always contain the current axis used
current_axis = contextvars.ContextVar("current_axis", default=None)


@contextmanager
def _temporary_environ(envs: dict[str, str]) -> Iterator[None]:
    """Override NCCL environment during group creation, this is to allow experimental configurations and can lead to undefined behavior.

    Enabled only by MAESTRO_USE_PER_AXIS_ENV_VARS; NCCL may cache these settings,
    so different values per axis may lead to undefined behavior.
    """
    if not envs:
        yield
        return

    if os.environ.get("MAESTRO_USE_PER_AXIS_ENV_VARS", "").lower() not in {"1", "true", "yes", "on"}:
        logger.warning("Ignoring axis environment overrides: MAESTRO_USE_PER_AXIS_ENV_VARS is disabled")
        yield
        return

    envs_before = os.environ.copy()
    os.environ.update(envs)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(envs_before)


class Axis(ABC):

    def __init__(self, groups: list[list[int]], name: str = "", envs: dict[str, str] | None = None, params: dict | None = None, *, dist_utils, allocator=None):
        """
        Axis is a set of teams created using the same pattern. Each axis executes blocks serially and can run independently of other axes
        For now, each rank must be in one group of the axis
        """
        self.dist_utils = dist_utils
        self.allocator = allocator

        self.groups = groups
        self.name = name
        self.envs = envs or {}

        with _temporary_environ(self.envs):
            self.my_group = self.dist_utils.create_axis(groups, params=params)

        self.my_group_ranks = None
        r = self.get_rank()
        for group in self.groups:
            if r in group:
                self.my_group_ranks = group
                break

        if not self.my_group_ranks:
            raise NotImplementedError(f"Rank {r} is not in any group of axis {self.name}")

        self.root_rank = min(self.my_group_ranks)

    def team_size(self):
        return len(self.my_group_ranks)

    def get_rank(self):
        return self.dist_utils.get_rank()

    def get_world_size(self):
        return self.dist_utils.get_world_size()

    @contextmanager
    def use_axis(self, block_current_stream=False) -> Iterator[None]:
        """Context manager to use the axis"""
        token = current_axis.set(self)
        try:
            with self._axis_ctx():
                yield
        finally:
            current_axis.reset(token)

    @abstractmethod
    def _axis_ctx(self) -> ContextManager[None]:
        """Wrapper around a context manager to use the axis"""
        pass

    def allocate(self, size, fill=0):
        if self.allocator is None:
            raise ValueError("This axis has no tensor allocator")
        return self.allocator.allocate(size, [self]).fill_(fill)

    @abstractmethod
    def complete_work(self, work):
        """Make subsequent work on this axis observe completion."""
        pass

    def get_process_group(self):
        return self.my_group


class TorchAxis(Axis):

    def __init__(self, groups: list[list[int]], name: str = "", envs: dict[str, str] | None = None, params: dict | None = None, *, dist_utils, allocator=None):

        super().__init__(groups, name=name, envs=envs, params=params,
                         dist_utils=dist_utils, allocator=allocator)
        self.stream = torch.cuda.Stream()

    def get_process_group(self) -> torch.distributed.ProcessGroup:
        return self.my_group

    @staticmethod
    def record_event(event: torch.cuda.Event):
        event.record()

    @staticmethod
    def create_event() -> torch.cuda.Event:
        return torch.cuda.Event()

    def _axis_ctx(self):
        return torch.cuda.stream(self.stream)

    def wait_event(self, event: torch.cuda.Event):
        self.stream.wait_event(event)

    def synchronize(self):
        self.stream.synchronize()

    @classmethod
    def synchronize_all(cls):
        torch.cuda.synchronize()

    def complete_work(self, work):
        work.block_current_stream()


class HostAxis(Axis):
    """A serial host queue; the host executor supplies concurrency across axes."""

    def _axis_ctx(self):
        return nullcontext()

    def complete_work(self, work):
        work.wait()
