## Tests and benchmarks for python wrappers and functions

This directory contains correctness tests and benchmarks, divided into subfolders:
1. `layer` — tests and benchmarks for key layers implemented in NNTile
2. `model` — tests and benchmarks for complete models and major subparts (e.g., blocks)
3. `loss` — tests and benchmarks for loss functions
4. `optimizer` — tests and benchmarks for optimizers
5. `nntile_core` — tests and benchmarks for core tensor operations and utilities

Common helpers and initializers are implemented as pytest fixtures in `nntile/wrappers/python/tests/conftest.py`. It also provides:
- `context`, `context_cuda` — runtime initialization and teardown
- `numpy_rng`, `torch_rng` — deterministic RNG setup (seed 42)
- `benchmark_operation`, `benchmark_model` — helpers built on `pytest-benchmark`
- CLI options and collection rules for dtype filtering and benchmark selection


### Correctness tests

Correctness tests use dtype-specific tolerances appropriate for the underlying implementation. Run tests from this directory or any subdirectory with `pytest`:

```
pytest
```

Run a single test file:
```
pytest nntile_core/test_tensor_add_inplace.py
```

Run a single test:
```
pytest nntile_core/test_tensor_add_inplace.py::test_add_inplace
```

Filter by keyword:
```
pytest -k add_inplace
```

### Benchmarks

Benchmarks use the same `pytest` CLI and can be run from any subdirectory. Benchmarks are disabled by default and are collected but skipped unless explicitly enabled with the `benchmark` marker:

```
pytest -m benchmark
```

Run benchmarks matching a keyword:
```
pytest -m benchmark -k test_add
```

### Dtype selection

Tests and benchmarks that parametrize `dtype` using the predefined set can be filtered with the `--dtype` option. Supported choices:
`fp32`, `fp16`, `bf16`, `fp32_fast_tf32`, `fp32_fast_fp16`, `fp32_fast_bf16`.

- Run only `bf16`:

```
pytest --dtype=bf16
```

- Run for multiple dtypes:

```
pytest --dtype=fp32 --dtype=fp16
```

By default, tests run for all implemented dtypes.
