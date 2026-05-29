# NNTile Python package

Build with CMake from the repository root:

```bash
cmake -S . -B build -DNNTILE_PRESET=full -DBUILD_PYTHON_WRAPPERS=ON
cmake --build build --target nntile_py
export PYTHONPATH=build/python
python -c "import nntile; print(nntile.NNGraph)"
```

The extension module is `nntile.nntile` (`.so` inside the package); `nntile/__init__.py`
re-exports the public API.
