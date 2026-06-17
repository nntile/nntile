import torch
import torch_nntile
from torch_nntile import _C


if not _C.has_libnntile():
    raise SystemExit("torch_nntile wheel was built without libnntile")

torch_nntile.init_context(ncpu=1, ncuda=0, verbose=0, cpu_fallback=False)
torch_nntile.restrict_cpu()

lhs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).to("nntile")
rhs = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32).to("nntile")
result = (lhs + rhs).cpu()

torch.testing.assert_close(
    result,
    torch.tensor([5.0, 7.0, 9.0], dtype=torch.float32),
)
