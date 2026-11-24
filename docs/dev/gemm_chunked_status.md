# Chunked GEMM: where the work stopped (2026-11-24)

Entry point for the native chunked fp32 GEMM family
(`nntile/src/kernel/gemm_chunked/`) on branch `gemm-chunked`. The work is
parked here. This note says what the family does, what it costs against
cuBLAS, why, what is verified, what is not merged, and what to do first
when picking it up again.

Measurements:

- [gemm_routes_chunk_class.md](gemm_routes_chunk_class.md): every GEMM of
  a GPT-2 step, S to XL, on 1, 2 and 4 MiB chunk classes. The current
  picture; it includes the `tails` piece.
- [gemm_chunked_gpt2_routes.md](gemm_chunked_gpt2_routes.md): 4 and 16 MiB
  engines, measured before `tails` existed (commit `12aa3564`).

## The result

Three routes on the same operands: contiguous cuBLAS, staging (cuBLAS on
a gathered copy, the copies included) and the chunked family in place on
the chunk tables.

| | GEMM time per GPT-2 step, ratio to contiguous cuBLAS |
|---|---|
| staging | 1.24–1.34× at 1024 tokens, 1.43–1.56× at 4096 tokens |
| chunked | 1.30–1.52× at 1024 tokens, 1.78–1.95× at 4096 tokens |

(Range over the four models and the three chunk classes.)

- **The chunked route is not an improvement over staging on any model.**
  With every row of the step on the native family, it is slower than
  staging everywhere.
- **One family competes.** On the six MLP rows of GPT-2 small, medium and
  large the chunked route takes 1.01–1.05× cuBLAS per model (except
  small at 1024 tokens on the 1 MiB and 2 MiB classes: 1.24×), and it is
  faster than staging on 100 of those 108 rows. Per row, 25 of 108 are
  within 2 % of cuBLAS.
- **Attention, GPT-2 XL and the lm head are lost.** Attention rows:
  2.4–2.9× cuBLAS against 1.8–2.1× for staging. XL: 1.20–1.54× against
  1.08–1.19×. lm head: 1.25–2.33× against 1.07–1.20×.
- **Over all 360 row-class pairs** the chunked route is faster than
  cuBLAS on 16, within 2 % on 35, within 10 % on 119, and faster than
  staging on 178.
- **Nothing in the trainer uses any of this.** Production routing is
  unchanged: the wrapper's ownership picker never consults the extension
  pieces. In a GPT-2 small
  run (4 MiB engine, no memory pressure) 9573 of 9603 GEMM calls took
  plain cuBLAS, 30 were staged and none ran natively. The chunked route
  runs through `launch_forced()` only: the benchmark, `FORCE_CHUNKED`
  and the workspace-overflow rescue.

A bar of "within 2 % of cuBLAS on each of the 72 MLP cases of S, M and L
on the 4 and 16 MiB engines" was set for this work and not met: 35 of 72
cases pass, 7 before (the second document above).

## Why the chunked route is behind

In order of size, each measured on this machine:

1. **The epilogue.** The bridge's epilogue resolves the chunk table for
   every output element with a 64-bit multiply and a division, and
   carries four unrolled guarded paths: about 8,400 instructions per
   thread against 1,130 in CUTLASS's own epilogue, with the same 512
   FMAs. About 7 % of a call at k = 768 and 2 % at k = 3072.
2. **The block barriers of the mainloop.** A 128x128x8 k-panel costs a
   256-thread threadblock 2,052 clocks in pure compute (the FFMA peak),
   about 2,150 clocks in CUTLASS's SIMT engine and 2,190 in the bridge on
   NN, 2,300 on NT. The difference is the two `__syncthreads()` and the
   stage store per panel: two barriers alone cost 197 clocks. Reading
   operands from shared memory and the chunk-table lookup cost nothing.
   Only one threadblock runs on an SM at a time (twice the grid takes
   twice the time), although the register count of the NN kernels (116)
   would fit two.
3. **The last wave.** A grid's time steps by one round at every multiple
   of 68 threadblocks. The worst MLP rows have 80 to 144 threadblocks,
   so up to half of the last wave is empty.
4. **Shapes without a tuned kernel.** XL and the lm head run the `tails`
   kernel, one general tile shape with predicated loads. `lm_head_dgrad`
   at 1024 tokens is 104 threadblocks over 6,283 k-panels and wants a
   split along k.

## What is on the branch

| part | serves | state |
|---|---|---|
| `bench_gemm_chunked_gpt2` | the benchmark, with a row filter | verified: every timed route is compared with cuBLAS before timing |
| `gemm_chunked_ext.*`, `ext/` | extension pieces: one translation unit per piece, asked by `launch_forced()` | builds and runs; pieces are reachable only through `launch_forced()` |
| `ext/flat.cu` | 12 all-flat MLP shapes (16 MiB engine) | the only piece a separate critic run passed (12 of 12 within 2 % of cuBLAS, two runs) |
| `ext/mixed.cu` | 30 mixed single-chunk/chunk-table MLP cases | critic run: 14 of 30 within 2 %, results correct on both engines, no out-of-scope regression |
| `ext/proj.cu` | projection rows at 4096 tokens | critic run: 4 of 8 within 2 %, clean |
| `ext/wgrad.cu` | weight gradients at 4096 tokens | builder's numbers only (3 of 8 within 2 %); its results were checked against cuBLAS by the benchmark on every timed row |
| `ext/large1024.cu` | GPT-2 large projection rows at 1024 tokens | 0 of 6 within 2 %; ratios 1.03–1.62× down to 1.02–1.08× |
| `ext/tails.cu`, tail mode in `gemm_chunked_bridge.cuh` | any batch-1 shape outside the 128-multiple law, and mixed layouts the core bridge declines | new; see below |
| `supports_forced()` | lets `launch_forced()` and the wrapper's forced/rescue gate admit piece shapes | `supports()`, the ownership law, is unchanged |

Verification of `tails`: the kernel test compares 27 shapes (64-wide
tails on every transpose, odd and sub-tile sizes, flat, mixed and
chunk-table operands) against a CPU reference on four chunk classes with
alpha 0.7 and beta 0.3, worst deviation 6.7e-6; the memory-access checker
reports no error on 15 shapes; the benchmark checks all 120 rows on
three classes against cuBLAS, worst relative error 1.2e-5. No critic run
and no tuning: its numbers are a first measurement.

The kernel test reaches the pieces only through `launch_forced()`; the
core's classic-family cases exercise the core. `test_gemm_chunked` passes
on the 1, 2 and 4 MiB engines.

## Not merged: work in progress

`gemm_chunked_wip/` holds the patches of two local branches, unverified
and not built into anything:

- `fc-*`: a hand-written SIMT mainloop for the fc rows (slower than the
  bridge on two of three model sizes) and two edits of the shared bridge
  header: quad runs keep their chunk base and re-read the table only at
  chunk crossings, the staged epilogue indexes by shift, and a direct
  epilogue from registers behind a defaulted parameter whose output
  mapping is wrong (left off).
- `large1024-*`: a multi-slice last-wave header (full-wave kernel on the
  bridge's text, job-table tail kernel, fixup), never routed.

They were made against the tree before `tails`; the header edits overlap
the tail-mode loaders and will not apply cleanly.

## For whoever continues

The leads, in the order the numbers put them:

1. A direct epilogue: accumulators to the output in 16-byte vectors, one
   chunk-table lookup per vector (the fc work above started it). It is
   the largest single cost on the k = 768–1280 rows.
2. A last wave that leaves no SM idle, on the bridge's own kernel text:
   the proj piece's cut wave reached 0.97× on one row; the large1024
   work generalizes it to several k slices.
3. Less mainloop bubble: deeper k panels, two resident threadblocks per
   SM (check residency by timing first: register counts suggested it,
   timing denied it), or one barrier per panel.
4. Split along k for the deep, narrow rows (`lm_head_dgrad`, the 1024-token
   projection rows of large and XL).

Things that cost time and are easy to repeat:

- Write pieces, not edits to `cuda.cu` or the bridge: the core
  translation unit takes 450 s to compile, a piece seconds (the tail
  mode was added to the bridge header as an opt-in template flag, which
  leaves every existing instantiation unchanged but still costs the
  450 s once).
- Do not clang-format `cuda.cu` or `gemm_chunked_bridge.cuh`: they are
  not formatted, and a whole-file pass rewrites thousands of lines.
- The bridge's mainloop body is at the FFMA ceiling (same instruction
  count as CUTLASS's engine). The attempts that wrote a mainloop from
  scratch ended slower than the bridge, by a few percent to 30 %.
- Measure with the benchmark, under the machine's GPU lock, with nothing
  else on the GPU; compare cuBLAS against itself between runs to size
  the noise (about 1 % per row).
- GPU performance counters are not available on this host (the driver
  restricts profiling to host administrators), so evidence comes from
  SASS and timing probes.

The rig that ran this (build containers per piece, the benchmark wrapper
with the GPU lock, the blind A/B scripts, every builder's and critic's
report) is outside the repository, on the machine the work ran on, under
`~/Code/nnhaul/.zcode/gauntlet-mlp/` and `~/Code/nnhaul/.zcode/chunk-classes/`;
their README files say how to resume.
