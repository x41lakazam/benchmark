from .main import cli
from .core.utils.distributed import nccl_torch_dist_utils

try:
    cli()
finally:
    nccl_torch_dist_utils.destroy()
