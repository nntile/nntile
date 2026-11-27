<!--
@copyright (c) 2026-present Skolkovo Institute of Science and Technology
                            (Skoltech), Russia. All rights reserved.

@file docs/chunk-model.md
The chunk model in one page: the one class, how a tile decomposes, what a
codelet receives, and how a COPY moves bytes. Standalone by design.

@version 1.0.0
-->

<!-- SPDX-License-Identifier: MIT -->

# The chunk model

NNHaul stores a tile as a chain of **cache chunks** drawn from the node's
pool, not as one contiguous block.

- **One class per process.** Every full chunk is exactly the same size, a
  power of two, read once from `NNHAUL_CHUNK_SIZE_MB` (default 4 MiB). A
  tile of `nbytes` decomposes into `ceil(nbytes / class)` chunks; the last
  chunk may be shorter (the *tail*), and no other size exists.
- **Residency is per chunk and per node.** The binding table maps, on each
  node, pool chunks to the logical chunks they hold, with one
  actual/outdated flag. A planner moves chunks between nodes as planned
  COPYs; the LRU planner is the default.
- **A COPY moves the chunk's occupied span** — the whole class for a full
  chunk, only the remainder for a tail. Byte-exact.
- **What a codelet receives.** A `Buffer` exposes `chunk_count()`,
  `chunk_bytes(i)`, `get_chunk_ptr<T>(i)`, and the chunk pointer table
  (`T const* const*`) that one-launch chunked kernels consume. A codelet
  therefore sees the chain, never a hidden whole-tile pointer.
- **Single-chunk case.** A tile no larger than the class is one chunk. A
  contiguous kernel may be launched on it; nothing about the model changes.
- **No unchunked mode.** There is no API to obtain a contiguous whole-tile
  allocation. NNTile's `NNHAUL_CHUNK_TILES=0` opt-out is rejected.

`<nnhaul/nnhaul.hh>` is the API; `docs/planner-api.md` covers planning over
these chunks. This page is complete on its own for using NNHaul; the
NNTile-side kernel decomposition is a separate contract.
