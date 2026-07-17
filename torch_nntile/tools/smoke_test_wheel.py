import torch

import torch_nntile
from torch_nntile import _C

# Smoke runs on CPU CI and CUDA wheels with ncuda=0; flag must be readable.
_ = torch_nntile.built_with_cuda()
if bool(_C.built_with_cuda()) != bool(torch_nntile.built_with_cuda()):
    raise SystemExit(
        "built_with_cuda mismatch between _build_info and _C "
        f"(py={torch_nntile.built_with_cuda()} native={_C.built_with_cuda()})"
    )

print(
    f"torch_nntile smoke: TORCH_NATIVE_OPS={torch_nntile.TORCH_NATIVE_OPS} "
    f"BUILT_WITH_CUDA={torch_nntile.built_with_cuda()}"
)
if not torch_nntile.TORCH_NATIVE_OPS:
    raise SystemExit(
        "expected NNTILE_TORCH_NATIVE_OPS wheel "
        f"(TORCH_NATIVE_OPS={torch_nntile.TORCH_NATIVE_OPS})"
    )

torch_nntile.init_context(ncpu=1, ncuda=0, verbose=0, cpu_fallback=False)
torch_nntile.restrict_cpu()

with torch.no_grad():
    lhs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).to("nntile")
    rhs = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32).to("nntile")
out = lhs + rhs
torch_nntile.compile_graph()
torch_nntile.run()
with torch.no_grad():
    result = out.cpu()

torch.testing.assert_close(
    result,
    torch.tensor([5.0, 7.0, 9.0], dtype=torch.float32),
)
