# Dirty C++ tests (CI)

In the **Build and test** workflow, the job **build and run dirty tests** only
builds and runs C++ tests that are **dirty** for the current change set.

## What “dirty” means

A test is dirty when the diff from the merge base to `HEAD` touches code that
can affect it, according to `.github/scripts/dirty-cpp-tests-lib.sh`:

| Change | Dirty tests |
|--------|-------------|
| `nntile/tests/<layer>/*.cc` | That test executable (`tests_<layer>_<op>`) |
| `nntile/src/kernel/...` | Kernel, StarPU, core tile, and graph tensor-op tests for that op |
| `nntile/src/starpu/...` | StarPU and above |
| `nntile/src/core/...` | Core tile and graph tensor-op tests |
| `nntile/src/tensor/ops/...` | `tests_graph_tensor_ops_*` |
| `nntile/src/tensor/<file>.cc` (not under `ops/`) | `tests_graph_tensor_*` |
| Graph headers/sources under `tile/`, `nn/`, `module/`, `io/`, `model/` | Matching `tests_graph_*` tests |
| Model `generate_test_data.py` (per family) | All block tests for that model family |
| CMake, `external/`, top-level headers, runtime, BLAS glue | **Full** C++ suite |
| Unmatched paths | **Full** C++ suite (safe default) |

`NotImplemented` tests are still excluded at run time (`ctest -LE NotImplemented`).

Model tests use CTest **fixtures** (`*_data_setup`); only the C++ executable
targets are built. CTest runs fixture scripts before the test when required.

## CI job flow

0. After `actions/checkout`, **Prepare git worktree in container** runs
   `.github/scripts/ci-ensure-git-worktree.sh` so `git diff` works inside the
   `ubuntu:24.04` container (the default checkout gitfile points at host paths).
1. Restore `libnntile` from **build libnntile**.
2. Configure with `BUILD_TESTS=ON` and model-fixture Python (`NNTILE_MODEL_PYTHON`).
3. **plan** — compute dirty targets from `origin/<base>..HEAD`.
4. **build** — `cmake --build` only dirty targets (or all tests if the full suite is dirty).
5. **run** — `ctest -R` with the same dirty set (or full suite).

On pull requests, the diff base is `github.event.pull_request.base.sha`
(`NNTILE_DIFF_BASE` in the workflow). On push or schedule, the job fetches
`origin/<DIFF_BASE>` (default `graph_api`) and uses `git merge-base` with
`HEAD`.

## Local use

```bash
# After a normal BUILD_TESTS=ON configure + build of libnntile:
.github/scripts/run-dirty-cpp-tests.sh my-branch graph_api

# Same logic as CI (plan / build / run):
.github/scripts/ci-dirty-cpp-tests.sh plan
.github/scripts/ci-dirty-cpp-tests.sh build
.github/scripts/ci-dirty-cpp-tests.sh run
```
