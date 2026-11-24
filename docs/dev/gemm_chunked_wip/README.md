# Unmerged work in progress: the chunked GEMM family

Patches of two local branches made on 2026-11-24 against the tree before
the `tails` piece (commits `3bab2071` and `2f5f8a27`). Nothing here is
built, tested or routed; they are kept so the ideas are not lost (as
`git format-patch` output, gzipped because patch context lines carry
trailing blanks: `gunzip -c <file> | git apply --check`). See
`../gemm_chunked_status.md`.

- `fc-1-piece-and-mainloop.patch.gz`: an `fc` piece with a hand-written
  two-stage SIMT mainloop and a fixed-order split tail. It claims only
  GPT-2 medium's fc rows at 4 MiB and is slower than the bridge on small
  and large. A probe of its mainloop showed that skipping the block
  barrier saves 6–10 %.
- `fc-2-bridge-quad-runs-keep-chunk-base.patch.gz`: the bridge's quad
  loaders keep the base of the chunk a run is in and re-read the table
  only after a chunk crossing (the panel's data load no longer waits on
  a table lookup); the staged epilogue computes its element index with a
  shift instead of a 64-bit multiply and divide. Unverified on chunk
  tables and on the 16 MiB engine.
- `fc-3-bridge-direct-epilogue-mapping-wrong.patch.gz`: a direct epilogue
  (accumulators straight to the output, 16-byte vectors, one table
  lookup per vector) behind a defaulted template parameter. Its output
  mapping is wrong; the parameter defaults to off.
- `large1024-1-wave-cut-header.patch.gz`: a last wave that leaves no SM
  idle: full waves run the bridge's kernel text, the tail tiles are cut
  into several k ranges on a job table, one fixed-order fixup. Its result
  check was broken and the tail and fixup kernels never ran.

The two bridge patches edit the loaders that now carry the opt-in tail
mode (`Tails` in `BridgeQuadA`/`BridgeQuadB`); expect conflicts.
