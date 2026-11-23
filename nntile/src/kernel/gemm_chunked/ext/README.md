# Extension pieces of the native chunked GEMM family

A piece is one translation unit, `nntile/src/kernel/gemm_chunked/ext/<name>.cu`,
that adds configurations to the family without touching `cuda.cu`. The
build registers every file in this directory by its name; a piece defines
three functions in `nntile::kernel::gemm_chunked::ext` (contract:
`../gemm_chunked_ext.hh`):

    int  <name>_pick(int base, const Call &call)
         the call's configuration id (`base + local id`) or -1 when the
         piece does not claim it; pieces are asked in file-name order and
         the first claim wins, so claim only the shapes and layouts you
         serve (the call carries the shape, the operand shifts and the
         strides);
    const char *<name>_name(int cfg)              the id's display name;
    bool <name>_launch(int cfg, const Call &call, cudaStream_t stream)
         run it; false without launching when the piece cannot take the
         call (the caller then falls back like for any core configuration).

Where the hooks sit (`../cuda.cu`): `launch_forced()` asks `ext::pick()`
before its shape rules, so a piece's claim is what it runs on that call.
`launch()` (production ownership) never consults the pieces except
through the `NNHAUL_GEMM_CHUNKED_CFG` pin. `launch_forced()` retries the
core bridge and split-K bridge only when the piece declined, never after
it ran.

Start from `TEMPLATE.cu.txt`: copy it to `ext/<name>.cu`, rename the
functions, fill in the pick. It compiles as it is (claiming nothing).

What a piece can reuse:

- `../gemm_chunked_bridge.cuh`: the chunk-table-aware CUTLASS 2.11 SIMT
  mainloop (`GemmBridge`, `BridgeQuadA/B`, `BridgeStoreA/B`,
  `ChunkedEpilogueIterator`, `GemmChunked`, `GemmChunkedSplitKParallel`).
  It relies on `<cutlass/gemm/device/gemm.h>` and
  `<cutlass/gemm/device/gemm_splitk_parallel.h>` being included first
  (the template does). The quad loaders are written for the 128x128x8
  threadblock tile and 256 threads.
- the CUTLASS 2.11 headers (on this directory's include path);
- `ext::scratch(bytes)`: a grow-only device scratch buffer (split-K
  partials);
- worked launchers for the bridge and its split-K sibling:
  `../cuda.cu`, `p7cut::run_bridge`, `p7cut::launch_bridge`,
  `p7cut::run_bridge_sk`, `p7cut::launch_bridge_sk`.

Only a piece's own translation unit recompiles when it changes.
