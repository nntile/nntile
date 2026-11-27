import os

import torch

import torch_nntile
from torch_nntile import _C
from torch_nntile.nn import functional, module
from torch_nntile.nn.model import DeepReLU

# Wheel smoke compiles a tiny graph; opt in to the public compiler.
os.environ.setdefault("NNTILE_ENABLE_LOCAL_COMPILER", "1")

# The build-info flag must agree with the loaded extension.
_ = torch_nntile.built_with_cuda()
if bool(_C.built_with_cuda()) != bool(torch_nntile.built_with_cuda()):
    raise SystemExit(
        "built_with_cuda mismatch between _build_info and _C "
        f"(py={torch_nntile.built_with_cuda()} native={_C.built_with_cuda()})"
    )

if not hasattr(functional, "relu") or not hasattr(module, "Linear"):
    raise SystemExit("expected torch_nntile.nn.functional and nn.module")
if DeepReLU is not torch_nntile.nn.model.DeepReLU:
    raise SystemExit("expected torch_nntile.nn.model.DeepReLU")

print(
    f"torch_nntile smoke: backend={torch_nntile.backend()} "
    f"BUILT_WITH_CUDA={torch_nntile.built_with_cuda()}"
)

# The wheel is the CUDA product: classic torch_nntile.nn ops on one device.
# A wheel-build runner without a GPU verifies the artifact and skips compute.
if not torch.cuda.is_available():
    print("torch_nntile smoke: no CUDA device; skipping compute leg")
    raise SystemExit(0)

torch_nntile.init_context(ncpu=0, ncuda=1, verbose=0, cpu_fallback=False)
with torch.no_grad():
    x = torch.tensor([-1.0, 2.0, -3.0], dtype=torch.float32).to("nntile")
out = functional.relu(x)
torch_nntile.compile_graph()
torch_nntile.run()
with torch.no_grad():
    result = out.cpu()
torch.testing.assert_close(
    result,
    torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32),
)
print("torch_nntile smoke: NNHaul classic relu OK")
