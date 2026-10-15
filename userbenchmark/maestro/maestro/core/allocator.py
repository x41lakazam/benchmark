"""Backend-owned tensor placement, dtype, and optional memory registration."""

import torch


class TorchAllocator:
    def __init__(self, device, dtype, register_buffers=False):
        if register_buffers:
            raise ValueError("This allocator does not support registered buffers")
        self.device = device
        self.dtype = dtype

    def allocate(self, size, axes=()):
        return torch.ones(size, device=self.device, dtype=self.dtype)

    def close(self):
        pass


class NCCLAllocator(TorchAllocator):
    def __init__(self, device, dtype, register_buffers=False):
        super().__init__(device, dtype)
        self.register_buffers = register_buffers
        self._pool = None
        self._groups = []

    def allocate(self, size, axes=()):
        if not self.register_buffers:
            return super().allocate(size, axes)
        groups = list(dict.fromkeys(axis.get_process_group() for axis in axes))
        if not groups:
            raise ValueError("Registered buffers require an NCCL process group")
        device = torch.device("cuda", torch.cuda.current_device())
        if self._pool is None:
            self._pool = torch.cuda.MemPool(groups[0]._get_backend(device).mem_allocator)
        with torch.cuda.use_mem_pool(self._pool):
            tensor = torch.ones(size, device=device, dtype=self.dtype)
        torch.cuda.synchronize()
        for group in groups:
            if group not in self._groups:
                group._get_backend(device).register_mem_pool(self._pool)
                self._groups.append(group)
        return tensor

    def close(self):
        if self._pool is None:
            return
        torch.cuda.synchronize()
        device = torch.device("cuda", torch.cuda.current_device())
        for group in self._groups:
            group._get_backend(device).deregister_mem_pool(self._pool)
        self._groups.clear()
        self._pool = None
