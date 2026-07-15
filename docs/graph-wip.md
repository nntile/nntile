# Graph API (historical)

The standalone **NNGraph** C++ stack (`nntile/nn`, `module`, `model`, …) and
the `python/nntile` bindings have been **removed**. Training and models now go
through **libtorch_nntile** (`device=nntile`) on top of **libnntile**
(TensorGraph → TileGraph → Runtime).

See:

- [libtorch_nntile migration](dev/libtorch_nntile_migration.md)
- [torch_nntile](torch_nntile.md)
- [C++ overview](cpp/README.md)
- O(N) compiler design: [graph_compiler_on_design.md](dev/graph_compiler_on_design.md)
