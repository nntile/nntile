# Plan: single NVIDIA stack for CUDA `torch_nntile` wheels

Goal: stop intermittent GitHub Actions disk failures by ensuring
**NVIDIA math/runtime libraries exist only once on disk** during the
manylinux CUDA wheel build, sourced from the same pip packages LibTorch
already depends on. Keep only the **minimal system pieces LibTorch cannot
provide** (`nvcc`, `libcuda` stub).

This is a design plan only; no implementation in this document.

## Problem

The `cp312-manylinux_x86_64` job stacks several multi‑GB CUDA copies:

| Layer | Source | Typical contents |
|-------|--------|------------------|
| A | `install_linux_cuda_toolkit.sh` (dnf) | Full CUDA 12.8 toolkit: nvcc, headers, **devel libs**, stubs |
| B | `setup_torch_cuda_env.sh` | `torch==2.9.1+cu128` + transitive `nvidia-*-cu12` + extra `nvidia-cudnn-cu12` |
| C | cibuildwheel `before-build` | torch cu128 + cudnn **again** |
| D | cibuildwheel `before-test` | torch **again** + six nvidia packages |
| E | StarPU / nntile / torch_nntile build trees | `/tmp` objects |

Peak usage is often download + extract + install (≈2–3× final size).
GitHub-hosted runners (~14 GB free) sit near the ceiling; torch/CUDA
wheels keep growing, so failures become more frequent.

## What we actually link (facts from tree)

### libnntile (`USE_CUDA=ON`)

From root `CMakeLists.txt` (`nntile_apply_cuda`):

- `CUDA::cublas` (pulls `cublasLt`)
- `CUDNN::cudnn_all` (cuDNN 9 split libs via cudnn-frontend’s `cuDNN.cmake`)
- Device link of ~67 `nntile/src/kernel/**/cuda.cu` → `CUDA::cudart`
- Public `StarPU::starpu` (CUDA StarPU also wants **`libcudart`** + **`libcuda`**)

**Not linked by NNTile CMake:** `cusparse`, `cusolver`, `cufft`, `nvrtc`.

cuDNN is always used when CUDA is on (`nntile/src/context.cc`:
`cudnnCreate`, `starpu_cublas_init`). Flash SDPA (`USE_FLASH_SDPA`) is
optional and currently off by default; it does not change the base
cublas/cudnn requirement.

### libtorch_nntile

- Links `nntile` (inherits CUDA) + `${TORCH_LIBRARIES}`
- C++ only; no `.cu` of its own
- Wheel repair (`tools/repair_wheel_linux.sh`) **excludes** torch/nvidia
  libs and RPATHs into `site-packages/nvidia/{cublas,cudnn,...}/lib`

### LibTorch packaging reality

- `torch/lib` has `libtorch_cuda.so`, `libc10_cuda.so`, … — **not**
  `libcublas` / `libcudnn` / `libcudart`
- Those live in **pip** `nvidia-*-cu12` next to torch (same install)
- So “link against LibTorch’s nvidia dependencies” means:
  **link against the pip `nvidia-*` packages that torch already installed**,
  not against files inside `torch/lib`

### What pip cannot replace today

| Need | Why | Who provides today |
|------|-----|--------------------|
| **nvcc** | `enable_language(CUDA)` + 67 `.cu` files | System toolkit (`cuda-minimal-build`) |
| **`libcuda.so` stub** | StarPU CUDA link | System `cuda-driver-devel` stubs |
| Real **driver** at runtime | StarPU CUDA workers | Host GPU driver (not in the wheel) |

`nvidia-cuda-nvcc-cu12` ships ptxas/CRT pieces, **not** a usable `nvcc`.

Headers for `cuda_runtime.h` / `cublas_v2.h` / `cudnn.h` **can** come from
pip `nvidia-cuda-runtime` / `nvidia-cublas` / `nvidia-cudnn` if include
paths are wired; they need not come from `/usr/local/cuda/include` once
cmake is pointed at pip.

## Target architecture

```text
┌─────────────────────────────────────────────────────────┐
│  Once per container: pip install torch+cu128            │
│  → brings nvidia-{cuda_runtime,cublas,cudnn,...}-cu12   │
│  SINGLE on-disk copy of all math/runtime NVIDIA .so’s   │
└─────────────────────────────────────────────────────────┘
        ▲ link / RPATH / CMAKE_LIBRARY_PATH / CUDNN_*
┌───────┴─────────────────────────────────────────────────┐
│  Minimal system “compiler kit” (no math .so’s)          │
│  • nvcc (+ CRT enough to compile .cu)                   │
│  • lib64/stubs/libcuda.so                               │
│  Optional: only headers not available from pip          │
└─────────────────────────────────────────────────────────┘
        ▲
┌───────┴─────────────────────────────────────────────────┐
│  StarPU --with-cuda-dir=<compiler kit>                  │
│  libnntile → cublas/cudnn/cudart from pip nvidia paths  │
│  libtorch_nntile → nntile + Torch                       │
│  auditwheel: exclude nvidia/torch; RPATH → nvidia/*/lib │
└─────────────────────────────────────────────────────────┘
```

**Invariant:** after `before-all`, `find`/`du` of `libcublas.so*` /
`libcudnn.so*` / `libcudart.so*` under the container should show
**only** paths under the active Python’s `site-packages/nvidia/…`
(plus build-tree copies of *our* libs). No second copy under
`/usr/local/cuda/lib64` for those sonames.

## Phased plan

### Phase 0 — Measure (short)

Add temporary `df -h` / `du -sh` checkpoints in CI (or a one-off workflow
dispatch) after:

1. dnf CUDA toolkit
2. pip torch cu128
3. StarPU + nntile build
4. before-build / before-test

Record which step crosses the cliff. Keep the logging behind a flag or
remove after the redesign ships.

### Phase 1 — Deduplicate pip torch (quick win, low risk)

**Still installs a system toolkit**, but stops reinstalling multi‑GB pip
stacks.

1. **`setup_torch_cuda_env.sh`**
   - One `pip install --no-cache-dir --index-url cu128 torch torchvision`
   - Drop redundant `pip install nvidia-cudnn-cu12` (already a torch dep)
   - `pip cache purge` after install
2. **`pyproject.toml` cibuildwheel linux hooks**
   - `before-build`: do **not** reinstall torch/cudnn if already present
     (detect `import torch` + CUDA build tag, or skip when
     `TORCH_NNTILE_USE_CUDA=1` and env points at before-all python)
   - `before-test`: same — reuse the before-all/cp312 env; only install
     missing *non-nvidia* test deps
3. Prefer cibuildwheel’s same CPython for before-all and the wheel
   (`wheel_python.sh` already targets cp312); avoid installing torch into
   a throwaway interpreter

**Success metric:** torch+cu128 downloaded/installed **once** per job.

### Phase 2 — Thin system toolkit; link math libs from pip (core goal)

**Principle:** system CUDA = compiler + driver stub only.

1. **Shrink `install_linux_cuda_toolkit.sh` package set**
   - Keep: something that provides **`nvcc`** and **`lib64/stubs/libcuda.so`**
     (today: `cuda-minimal-build-*`, `cuda-driver-devel-*`)
   - Drop (or stop linking against): `cuda-libraries-devel-*`,
     `cuda-nvrtc-devel-*` once cmake no longer needs their `.so` files
   - After install: `dnf clean all`; delete unused toolkit lib copies if
     the meta-packages still pull them (document residual size)
2. **Point CMake at pip nvidia for link/includes**
   - After torch install, derive:
     - `NVIDIA_CUBLAS_ROOT` / include+lib from `nvidia.cublas`
     - `CUDNN_PATH` / include+lib from `nvidia.cudnn` (already partially done)
     - `CUDA_RUNTIME` include/lib from `nvidia.cuda_runtime` if needed
   - Pass into nntile configure:
     - `CUDAToolkit_ROOT` may still be the **thin** `/usr/local/cuda` for
       nvcc discovery
     - Force library search so `CUDA::cublas` / `find_library(cudart)`
       resolve to **pip** paths (e.g. `CMAKE_LIBRARY_PATH`,
       `CMAKE_PREFIX_PATH` order, or custom imported targets)
   - **Hard requirement:** linked `NEEDED` entries on `libnntile.so` for
     `libcublas` / `libcudnn*` / `libcudart` must match pip SONAMEs
     (verify with `readelf -d`), not toolkit copies
3. **StarPU**
   - Keep `--with-cuda-dir` on the thin toolkit (needs stubs + cudart
     headers for its build)
   - Prefer StarPU linking cudart from the same pip tree if StarPU’s
     configure allows explicit lib paths; otherwise accept StarPU
     temporarily linking toolkit cudart **only if** that toolkit cudart
     is not a second full copy — ideally StarPU also uses pip cudart
4. **Wheel metadata**
   - Keep declaring the nvidia packages torch already needs (or rely on
     torch’s deps alone and trim `_linux_nvidia_requires` to what
     `libnntile` RPATH actually needs: at least
     `cuda_runtime`, `cublas`, `cudnn`)
   - `cusparse` / `cusolver` / `nvjitlink` are **not** NNTile link deps;
     keep them only if torch or auditwheel RPATH still requires them for
     `libtorch_cuda` resolution — prefer inheriting via `torch` dependency
     rather than duplicating pins

**Success metric:** `du` of `/usr/local/cuda/lib64` has no
`libcublas`/`libcudnn` (or they are unused stubs); only one tree of math
libs under `site-packages/nvidia`.

### Phase 3 — CMake support for “pip CUDAToolkit” (repo quality)

Encode Phase 2 so local and CI builds share one path:

1. Helper (e.g. `cmake/NNTileFindPipCuda.cmake` or extend wheel scripts +
   documented cache vars) that:
   - Locates pip nvidia packages via Python
   - Sets `CUDNN_*`, cublas include/lib, optional `CUDAToolkit` hints
2. Document in `docs/build/README.md` / `torch_nntile/README.md`:
   - Dev machine: install torch cu128 first, then thin toolkit OR full
     toolkit; prefer pip libs when both exist
3. Optional CI assert job: fail if `ldd libnntile.so` resolves cublas/cudnn
   to `/usr/local/cuda` when `NNTILE_CUDA_FROM_PIP=1`

### Phase 4 — Optional longer-term (only if still tight)

| Idea | Effect | Cost |
|------|--------|------|
| Prebaked manylinux image with thin toolkit + torch cu128 | Eliminates download/extract peak every PR | Image maintenance |
| Cache StarPU + libnntile prefix artifact; cibuildwheel only builds `_C` | Less `/tmp` build tree | Cache invalidation |
| Restrict CUDA wheels to `workflow_dispatch` / tags | Fewer PR failures | Less PR signal |
| clang-cuda / precompiled fatbins to drop nvcc | Removes system toolkit entirely | Large engineering |

Do not block Phase 1–2 on these.

## Explicit non-goals / non-feasible claims

- **Linking only `torch/lib` for cublas/cudnn:** impossible; those SO files
  are not there.
- **Zero system CUDA packages with current `.cu` + StarPU CUDA:** not
  feasible without replacing nvcc and the driver stub story.
- **Bundling nvidia libs inside the wheel:** rejected; current
  auditwheel+RPATH+pip deps model is correct for size and license
  surface. “Once on disk” means once in the **build/test environment**
  and once in the **user’s** site-packages (via torch), not inside
  our wheel.

## Mapping: libnntile extras vs torch nvidia set

| Library | libnntile needs? | In torch cu128 deps? | Action |
|---------|------------------|----------------------|--------|
| cudart | Yes (link) | Yes (`nvidia-cuda-runtime`) | Use pip only |
| cublas / cublasLt | Yes | Yes (`nvidia-cublas`) | Use pip only |
| cudnn* | Yes | Yes (`nvidia-cudnn`) | Use pip only |
| cusparse | No (NNTile) | Yes (torch) | Do not install twice; rely on torch |
| cusolver | No (NNTile) | Yes (torch) | Same |
| nvjitlink | Transitive (cublas/torch) | Yes | Same |
| nvrtc / cufft / nccl / … | No (NNTile) | Yes (torch) | Do not add for NNTile |
| nvcc | Yes (build) | **No** | Keep thin system package |
| libcuda stub | Yes (link StarPU) | **No** | Keep `cuda-driver-devel` stubs |
| libcuda.so.1 | Runtime only | Driver | Host |

**Conclusion:** libnntile does **not** need nvidia packages beyond what
torch already pulls for math libs. The only *additional* build-time
nvidia-related pieces are **nvcc + libcuda stub**, which are not
duplicate math libraries if the toolkit is thinned.

## Suggested implementation order

1. Phase 1 (dedupe pip) — unblocks most CI flakes quickly  
2. Phase 2 (thin toolkit + pip link) — enforces single nvidia math stack  
3. Phase 3 (cmake helper + docs + CI assert) — makes it durable  
4. Phase 4 only if disk still fails

## Validation checklist (when implementing)

- [ ] One `pip install` of torch+cu128 per manylinux job  
- [ ] `pip cache purge` / `--no-cache-dir` around large installs  
- [ ] `readelf -d libnntile.so` → cublas/cudnn/cudart from pip paths  
- [ ] `ldd` under smoke test resolves nvidia libs under `site-packages/nvidia`  
- [ ] Wheel smoke (`tools/smoke_test_wheel.py`) still passes  
- [ ] StarPU CUDA init works with stub at link and driver at run  
- [ ] Disk: `df -h` after before-all stays comfortably under runner limit  

## Related files

- `torch_nntile/tools/install_linux_cuda_toolkit.sh`
- `torch_nntile/tools/setup_torch_cuda_env.sh`
- `torch_nntile/tools/build_wheel_deps.sh`
- `torch_nntile/tools/repair_wheel_linux.sh`
- `torch_nntile/pyproject.toml` (`[tool.cibuildwheel.linux]`)
- `torch_nntile/setup.py` / `_cuda_deps.py` (runtime requires)
- Root `CMakeLists.txt` (`USE_CUDA`, `nntile_apply_cuda`)
- `.github/workflows/torch-nntile-wheels.yml`
