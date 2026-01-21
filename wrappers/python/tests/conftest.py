# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/conftest.py
# Common PyTest configurations for testing
#
# @version 1.1.0

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest


def _ensure_starpu_home() -> None:
    """StarPU needs a writable $HOME/.starpu directory for calibration."""
    home = Path.home()
    try:
        starpu_dir = home / '.starpu'
        starpu_dir.mkdir(parents=True, exist_ok=True)
        test_file = starpu_dir / '.write_test'
        test_file.write_text('')
        test_file.unlink()
    except OSError:
        fallback = Path(tempfile.mkdtemp(prefix='starpu-home-', dir=Path.cwd()))
        os.environ['HOME'] = str(fallback)
        starpu_dir = fallback / '.starpu'
        starpu_dir.mkdir(parents=True, exist_ok=True)


_ensure_starpu_home()

import nntile


@pytest.fixture(scope='session')
def context() -> Generator[None, None, None]:
    context = nntile.Context(ncpu=1, ncuda=1, ooc=0, logger=0, verbose=0)
    context.restrict_cpu()
    try:
        yield None
    finally:
        nntile.starpu.wait_for_all()
        context.shutdown()


@pytest.fixture(scope='session')
def context_cuda() -> Generator[None, None, None]:
    context = nntile.Context(ncpu=1, ncuda=1, ooc=0, logger=0, verbose=0)
    context.restrict_cuda()
    try:
        yield None
    finally:
        nntile.starpu.wait_for_all()
        context.shutdown()


@pytest.fixture(scope='function')
def numpy_rng():
    bits = np.random.MT19937(42)
    return np.random.Generator(bits)


@pytest.fixture(scope='function')
def torch_rng():
    import torch
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


@pytest.fixture(scope='function')
def benchmark_operation(benchmark):
    def bench_fn(fn):
        return benchmark.pedantic(fn, iterations=5, rounds=10, warmup_rounds=5)
    return bench_fn


@pytest.fixture(scope='function')
def benchmark_model(benchmark):
    def bench_fn(fn):
        return benchmark.pedantic(fn, iterations=5, rounds=3, warmup_rounds=1)
    return bench_fn


def pytest_collection_modifyitems(config, items):
    # If the user asked for benchmarks (e.g., `-m benchmark`), don't skip them
    markexpr = config.getoption("-m") or ""
    is_benchmark_run = "benchmark" in markexpr
    # Otherwise, skip every test that has the "benchmark" mark
    if not is_benchmark_run:
        skip_bench = pytest.mark.skip(
            reason="Benchmark disabled. Run with: pytest -m benchmark"
        )
        for item in items:
            if "benchmark" in item.keywords:
                item.add_marker(skip_bench)

    # Apply --dtype filtering to any parametrized
    # tests or benchmarks that include "dtype"
    selected = config.getoption("dtypes")
    if selected:
        allowed = set(selected)
        skip_unselected = pytest.mark.skip(
            reason="Filtered out by --dtype selection"
        )
        for item in items:
            callspec = getattr(item, "callspec", None)
            if callspec and "dtype" in callspec.params:
                dtype_val = callspec.params["dtype"]
                if dtype_val not in allowed:
                    item.add_marker(skip_unselected)


ALL_DTYPES = [
    'fp32',
    'fp16',
    'bf16',
    'fp32_fast_tf32',
    'fp32_fast_fp16',
    'fp32_fast_bf16',
]


def pytest_addoption(parser):
    parser.addoption(
        "--dtype",
        action="append",
        dest="dtypes",
        choices=ALL_DTYPES,
        help="Only run tests for the given dtype(s). "
             "Repeat the option to include multiple, "
             "e.g. --dtype=bf16 --dtype=fp32.",
    )
