# NNTile Python package (`python/`)

Python bindings for the **libnntile** graph API (`NNGraph`, `Runtime`, models, optimizers).

## Build (required)

The extension links against `libnntile` and StarPU. Build from the repository root:

```bash
cmake -S . -B build -GNinja \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_C_COMPILER=gcc \
  -DUSE_CUDA=OFF \
  -DBUILD_PYTHON_WRAPPERS=ON
cmake --build build --target nntile_py
```

## Run

```bash
export PYTHONPATH="$(pwd)/build/python"
export LD_LIBRARY_PATH="$(pwd)/build/nntile:/opt/starpu/lib:${LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="/opt/starpu/lib/pkgconfig:${PKG_CONFIG_PATH}"

python3 -c "import nntile; print(nntile.NNGraph)"
python3 python/examples/mlp_example.py
```

## Tests

```bash
pip install pytest numpy
pytest python/tests -vv
```

## Examples

See [examples/README.md](examples/README.md).

## Install into a venv (optional)

After CMake build:

```bash
pip install build/python
```

You still need `LD_LIBRARY_PATH` for StarPU and `libnntile.so` unless installed system-wide.

## Future: `pip install` via scikit-build-core

A single-command install (`pip install ./python` driving CMake) is planned as a follow-up
using [scikit-build-core](https://scikit-build-core.readthedocs.io/). For now, CMake + `PYTHONPATH` is the supported path.
