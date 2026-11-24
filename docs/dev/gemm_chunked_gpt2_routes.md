# GPT-2 GEMM routes: cuBLAS vs workspace-staged vs native chunked

Baseline for the chunked-GEMM work: what each GEMM route of the nnhaul gemm
wrapper (`nntile/src/nnhaul/gemm.cc`) costs on the shapes a GPT-2 training
step issues, for the native family in `nntile/src/kernel/gemm_chunked/`.
Measured with `bench_gemm_chunked_gpt2`
(`nntile/tests/kernel/gemm_chunked_gpt2_bench.cc`) on 2026-11-24, after
the first optimization round described below; the 2026-11-23 baseline is
commit `f3c1cd03`. Every table below is the output of
`torch_nntile/tools/render_gemm_chunked_gpt2_tables.py` on the three CSV
files committed next to this doc.

**Routes.**

1. **cuBLAS** — plain `kernel::cublas::gemm` on flat contiguous operands.
2. **staged** — the wrapper's workspace-staged body: every multi-chunk
   operand is gathered into a contiguous workspace slot
   (`copy::chunked_segments`), cuBLAS runs on the slots, and a multi-chunk
   C is scattered back (`copy::chunked`). Single-chunk operands are not
   staged. Bitwise equal to cuBLAS.
3. **native** — the native chunked family in place on the chunk tables, in
   its best configuration for the shape (`launch_forced`), regardless of
   production routing. The configurations are the core's (`cuda.cu`) and
   the extension pieces under `kernel/gemm_chunked/ext/` (`ext/README.md`),
   which `launch_forced` asks first. `n/a` when the family cannot run the
   row: the shape is outside its law (the 50257-wide lm head, GPT-2 XL's
   H=1600).
4. **production** — what the wrapper does with the layout when no
   `NNHAUL_FORCE_CHUNKED_KERNELS` is set: plain cuBLAS when every operand
   is one chunk, else the native configuration the ownership picker
   selects, else staged (with the CUTLASS flat engine as the GEMM leg on
   the batch-1 rows of its winners table).

**Rows.** The 15 distinct GEMM shapes of one training step, forward and
backward: Q/K/V and O projections (`qkvO_*`, four calls per layer), the
three batched attention shapes (two calls per layer each), the six MLP
shapes and the three lm-head shapes. `tokens` is batch × seq per step;
`calls` is how often the step issues the row.

**Layout.** Operands are laid out like the engine's chunked tiles: a
multi-chunk operand is a table of whole-class chunks with a shorter last
one, and is treated as scattered; a single-chunk operand is one flat
allocation. The per-launch chunk-pointer table upload of the wrapper is
not part of the timed region.

**Conditions.** NVIDIA GeForce RTX 2080 Ti (sm_75), CUDA 12.4, cuBLAS
12.4.5.8, CUTLASS 2.11.0; one process on the GPU, SM clock 1335–1350 MHz,
at most 75 °C. Median of at least 50 timed iterations (CUDA event pairs,
five interleaved rounds). Every route was checked against the cuBLAS result
on the same data before timing: staged is bitwise identical, the native and
CUTLASS routes are within 3.6e-6 relative.

Two engine builds were measured, because the staged body's copy kernels are
compiled for one chunk class (rule 7): a 4 MiB-class nnhaul engine (the
library default) and a 16 MiB-class engine. The native family takes any
power-of-two class at runtime, so the 1, 2 and 8 MiB classes have native
numbers only. Step totals are sums of calls × median: GEMM time only.


## GEMM time per training step

| model | tokens/step | cuBLAS ms | 4 MiB: staged · native · production | 16 MiB: staged · native · production |
|---|---:|---:|---|---|
| small (H=768, 12 heads, 12 layers) | 1024 | 92.5 | 116.3 (1.25×) · 125.4 (1.35×) · 116.2 (1.25×) | 109.6 (1.19×) · 123.6 (1.34×) · 109.6 (1.19×) |
| small (H=768, 12 heads, 12 layers) | 4096 | 490.7 | 768.1 (1.56×) · 940.3 (1.92×) · 720.1 (1.47×) | 726.8 (1.48×) · 927.2 (1.89×) · 699.0 (1.42×) |
| medium (H=1024, 16 heads, 24 layers) | 1024 | 247.1 | 307.3 (1.24×) · 315.3 (1.27×) · 300.7 (1.21×) | 286.7 (1.17×) · 312.5 (1.27×) · 286.6 (1.17×) |
| medium (H=1024, 16 heads, 24 layers) | 4096 | 1360.5 | 2077.6 (1.53×) · 2492.4 (1.83×) · 1927.0 (1.42×) | 1971.9 (1.45×) · 2456.2 (1.80×) · 1891.2 (1.39×) |
| large (H=1280, 20 heads, 36 layers) | 1024 | 523.4 | 679.7 (1.30×) · 728.8 (1.39×) · 660.5 (1.26×) | 631.7 (1.21×) · 706.6 (1.35×) · 630.7 (1.21×) |
| large (H=1280, 20 heads, 36 layers) | 4096 | 2754.3 | 4110.2 (1.49×) · 4865.4 (1.77×) · 3798.6 (1.38×) | 4096.6 (1.49×) · 4882.0 (1.77×) · 3781.9 (1.37×) |
| xl (H=1600, 25 heads, 48 layers) | 1024 | 1048.6 | 1321.8 (1.26×) · 1365.8 (1.30×) · 1321.8 (1.26×) | 1240.2 (1.18×) · 1300.8 (1.24×) · 1239.9 (1.18×) |
| xl (H=1600, 25 heads, 48 layers) | 4096 | 5263.7 | 7525.6 (1.43×) · 8906.8 (1.69×) · 7113.6 (1.35×) | 7497.4 (1.43×) · 8884.7 (1.69×) · 7087.0 (1.35×) |

## Native family alone at the other classes

| model | tokens/step | cuBLAS ms | 1 MiB class: native, rows launched | 2 MiB class: native, rows launched | 8 MiB class: native, rows launched |
|---|---:|---:|---|---|---|
| small | 1024 | 92.4 | 129.5 (1.40×), 12/15 | 129.4 (1.40×), 12/15 | 121.0 (1.32×), 12/15 |
| small | 4096 | 491.0 | 921.5 (1.88×), 12/15 | 921.4 (1.88×), 12/15 | 918.1 (1.87×), 10/15 |
| medium | 1024 | 247.1 | 316.8 (1.28×), 12/15 | 317.0 (1.28×), 12/15 | 309.6 (1.26×), 12/15 |
| medium | 4096 | 1359.1 | 2461.8 (1.81×), 12/15 | 2462.2 (1.81×), 12/15 | 2453.2 (1.81×), 10/15 |
| large | 1024 | 522.8 | 722.0 (1.38×), 12/15 | 722.0 (1.38×), 12/15 | 700.1 (1.34×), 12/15 |
| large | 4096 | 2756.6 | 4845.1 (1.76×), 12/15 | 4847.7 (1.76×), 12/15 | 4823.2 (1.75×), 10/15 |
| xl | 1024 | 1050.3 | 1227.3 (1.17×), 3/15 | 1229.1 (1.16×), 3/15 | 1221.8 (1.16×), 3/15 |
| xl | 4096 | 5267.3 | 8568.5 (1.63×), 3/15 | 8564.7 (1.63×), 3/15 | 8563.6 (1.63×), 3/15 |

## Per row, 1024 tokens/step


**GPT-2 small (H=768, 12 heads, 12 layers), 1024 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 4 MiB | native 4 MiB | staged 16 MiB | native 16 MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 1024, 768, 768 × 1 | 48 | 158 | 160 (1.01×) | 171 (1.08×) | 156 (0.99×) | 171 (1.08×) |
| `qkvO_fwd` | 1024, 768, 768 × 1 | 48 | 136 | 137 (1.01×) | 174 (1.28×) | 135 (0.99×) | 174 (1.27×) |
| `qkvO_wgrad` | 768, 768, 1024 × 1 | 48 | 136 | 136 (1.00×) | 217 (1.60×) | 137 (1.01×) | 217 (1.60×) |
| `attn_ctx` | 1024, 64, 1024 × 12 | 24 | 234 | 420 (1.79×) | 385 (1.64×) | 417 (1.78×) | 385 (1.64×) |
| `attn_dKV` | 1024, 64, 1024 × 12 | 24 | 217 | 404 (1.86×) | 309 (1.43×) | 401 (1.85×) | 308 (1.42×) |
| `attn_scores` | 1024, 1024, 64 × 12 | 24 | 225 | 414 (1.84×) | 889 (3.96×) | 413 (1.84×) | 890 (3.96×) |
| `mlp_fc_dgrad` | 1024, 3072, 768 × 1 | 12 | 462 | 547 (1.19×) | 477 (1.03×) | 462 (1.00×) | 458 (0.99×) |
| `mlp_fc_fwd` | 1024, 3072, 768 × 1 | 12 | 472 | 563 (1.19×) | 483 (1.02×) | 470 (0.99×) | 462 (0.98×) |
| `mlp_fc_wgrad` | 768, 3072, 1024 × 1 | 12 | 507 | 590 (1.16×) | 479 (0.94×) | 506 (1.00×) | 462 (0.91×) |
| `mlp_proj_dgrad` | 1024, 768, 3072 × 1 | 12 | 466 | 554 (1.19×) | 492 (1.05×) | 463 (0.99×) | 461 (0.99×) |
| `mlp_proj_fwd` | 1024, 768, 3072 × 1 | 12 | 496 | 583 (1.17×) | 522 (1.05×) | 493 (0.99×) | 475 (0.96×) |
| `mlp_proj_wgrad` | 3072, 768, 1024 × 1 | 12 | 460 | 548 (1.19×) | 481 (1.05×) | 461 (1.00×) | 461 (1.00×) |
| `lm_head_dgrad` | 1024, 768, 50257 × 1 | 1 | 7182 | 8519 (1.19×) | n/a | 8517 (1.19×) | n/a |
| `lm_head_fwd` | 1024, 50257, 768 × 1 | 1 | 7219 | 8515 (1.18×) | n/a | 8575 (1.19×) | n/a |
| `lm_head_wgrad` | 50257, 768, 1024 × 1 | 1 | 6864 | 8190 (1.19×) | n/a | 8187 (1.19×) | n/a |
| **per training step (ms)** | | | **92.5** | **116.3 (1.26×)** | **125.4 (1.36×)** | **109.6 (1.19×)** | **123.6 (1.34×)** |

Production routes natively at 4 MiB: none.

Production routes natively at 16 MiB: none.

**GPT-2 medium (H=1024, 16 heads, 24 layers), 1024 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 4 MiB | native 4 MiB | staged 16 MiB | native 16 MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 1024, 1024, 1024 × 1 | 96 | 219 | 221 (1.01×) | 221 (1.01×) | 217 (0.99×) | 220 (1.01×) |
| `qkvO_fwd` | 1024, 1024, 1024 × 1 | 96 | 236 | 238 (1.01×) | 225 (0.96×) | 233 (0.99×) | 225 (0.95×) |
| `qkvO_wgrad` | 1024, 1024, 1024 × 1 | 96 | 216 | 218 (1.01×) | 218 (1.01×) | 214 (0.99×) | 218 (1.01×) |
| `attn_ctx` | 1024, 64, 1024 × 16 | 48 | 256 | 512 (2.00×) | 500 (1.95×) | 508 (1.98×) | 499 (1.95×) |
| `attn_dKV` | 1024, 64, 1024 × 16 | 48 | 222 | 474 (2.14×) | 401 (1.81×) | 469 (2.11×) | 399 (1.80×) |
| `attn_scores` | 1024, 1024, 64 × 16 | 48 | 296 | 547 (1.85×) | 1178 (3.99×) | 545 (1.85×) | 1177 (3.98×) |
| `mlp_fc_dgrad` | 1024, 4096, 1024 × 1 | 24 | 798 | 926 (1.16×) | 836 (1.05×) | 799 (1.00×) | 806 (1.01×) |
| `mlp_fc_fwd` | 1024, 4096, 1024 × 1 | 24 | 822 | 955 (1.16×) | 848 (1.03×) | 818 (1.00×) | 816 (0.99×) |
| `mlp_fc_wgrad` | 1024, 4096, 1024 × 1 | 24 | 788 | 916 (1.16×) | 821 (1.04×) | 788 (1.00×) | 800 (1.01×) |
| `mlp_proj_dgrad` | 1024, 1024, 4096 × 1 | 24 | 815 | 945 (1.16×) | 805 (0.99×) | 810 (0.99×) | 816 (1.00×) |
| `mlp_proj_fwd` | 1024, 1024, 4096 × 1 | 24 | 880 | 1011 (1.15×) | 825 (0.94×) | 874 (0.99×) | 821 (0.93×) |
| `mlp_proj_wgrad` | 4096, 1024, 1024 × 1 | 24 | 788 | 918 (1.16×) | 824 (1.05×) | 788 (1.00×) | 800 (1.02×) |
| `lm_head_dgrad` | 1024, 1024, 50257 × 1 | 1 | 9582 | 11089 (1.16×) | n/a | 11088 (1.16×) | n/a |
| `lm_head_fwd` | 1024, 50257, 1024 × 1 | 1 | 9523 | 11010 (1.16×) | n/a | 11025 (1.16×) | n/a |
| `lm_head_wgrad` | 50257, 1024, 1024 × 1 | 1 | 9115 | 10628 (1.17×) | n/a | 10626 (1.17×) | n/a |
| **per training step (ms)** | | | **247.1** | **307.3 (1.24×)** | **315.3 (1.28×)** | **286.7 (1.16×)** | **312.5 (1.27×)** |

Production routes natively at 4 MiB: `mlp_proj_dgrad`, `mlp_proj_fwd`.

Production routes natively at 16 MiB: none.

**GPT-2 large (H=1280, 20 heads, 36 layers), 1024 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 4 MiB | native 4 MiB | staged 16 MiB | native 16 MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 1024, 1280, 1280 × 1 | 144 | 328 | 394 (1.20×) | 526 (1.60×) | 328 (1.00×) | 514 (1.57×) |
| `qkvO_fwd` | 1024, 1280, 1280 × 1 | 144 | 353 | 418 (1.18×) | 543 (1.54×) | 353 (1.00×) | 540 (1.53×) |
| `qkvO_wgrad` | 1280, 1280, 1024 × 1 | 144 | 316 | 387 (1.23×) | 434 (1.37×) | 315 (1.00×) | 407 (1.29×) |
| `attn_ctx` | 1024, 64, 1024 × 20 | 72 | 358 | 720 (2.01×) | 634 (1.77×) | 656 (1.83×) | 616 (1.72×) |
| `attn_dKV` | 1024, 64, 1024 × 20 | 72 | 316 | 687 (2.18×) | 504 (1.60×) | 621 (1.97×) | 493 (1.56×) |
| `attn_scores` | 1024, 1024, 64 × 20 | 72 | 366 | 720 (1.97×) | 1473 (4.02×) | 678 (1.85×) | 1468 (4.01×) |
| `mlp_fc_dgrad` | 1024, 5120, 1280 × 1 | 36 | 1240 | 1442 (1.16×) | 1276 (1.03×) | 1418 (1.14×) | 1269 (1.02×) |
| `mlp_fc_fwd` | 1024, 5120, 1280 × 1 | 36 | 1281 | 1484 (1.16×) | 1348 (1.05×) | 1454 (1.13×) | 1294 (1.01×) |
| `mlp_fc_wgrad` | 1280, 5120, 1024 × 1 | 36 | 1197 | 1408 (1.18×) | 1294 (1.08×) | 1368 (1.14×) | 1229 (1.03×) |
| `mlp_proj_dgrad` | 1024, 1280, 5120 × 1 | 36 | 1250 | 1443 (1.15×) | 1307 (1.05×) | 1426 (1.14×) | 1226 (0.98×) |
| `mlp_proj_fwd` | 1024, 1280, 5120 × 1 | 36 | 1345 | 1543 (1.15×) | 1376 (1.02×) | 1516 (1.13×) | 1276 (0.95×) |
| `mlp_proj_wgrad` | 5120, 1280, 1024 × 1 | 36 | 1190 | 1396 (1.17×) | 1292 (1.09×) | 1368 (1.15×) | 1224 (1.03×) |
| `lm_head_dgrad` | 1024, 1280, 50257 × 1 | 1 | 11496 | 13229 (1.15×) | n/a | 13208 (1.15×) | n/a |
| `lm_head_fwd` | 1024, 50257, 1280 × 1 | 1 | 11986 | 13773 (1.15×) | n/a | 13674 (1.14×) | n/a |
| `lm_head_wgrad` | 50257, 1280, 1024 × 1 | 1 | 11366 | 13147 (1.16×) | n/a | 13067 (1.15×) | n/a |
| **per training step (ms)** | | | **523.4** | **679.7 (1.30×)** | **728.8 (1.39×)** | **631.7 (1.21×)** | **706.6 (1.35×)** |

Production routes natively at 4 MiB: `mlp_fc_dgrad`, `mlp_fc_fwd`, `mlp_fc_wgrad`, `mlp_proj_wgrad`.

Production routes natively at 16 MiB: none.

**GPT-2 xl (H=1600, 25 heads, 48 layers), 1024 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 4 MiB | native 4 MiB | staged 16 MiB | native 16 MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 1024, 1600, 1600 × 1 | 192 | 517 | 606 (1.17×) | n/a | 517 (1.00×) | n/a |
| `qkvO_fwd` | 1024, 1600, 1600 × 1 | 192 | 631 | 721 (1.14×) | n/a | 632 (1.00×) | n/a |
| `qkvO_wgrad` | 1600, 1600, 1024 × 1 | 192 | 500 | 594 (1.19×) | n/a | 499 (1.00×) | n/a |
| `attn_ctx` | 1024, 64, 1024 × 25 | 96 | 470 | 961 (2.05×) | 756 (1.61×) | 886 (1.89×) | 735 (1.56×) |
| `attn_dKV` | 1024, 64, 1024 × 25 | 96 | 413 | 875 (2.12×) | 601 (1.45×) | 795 (1.92×) | 588 (1.42×) |
| `attn_scores` | 1024, 1024, 64 × 25 | 96 | 455 | 895 (1.97×) | 1831 (4.03×) | 843 (1.85×) | 1833 (4.03×) |
| `mlp_fc_dgrad` | 1024, 6400, 1600 × 1 | 48 | 1866 | 2157 (1.16×) | n/a | 2114 (1.13×) | n/a |
| `mlp_fc_fwd` | 1024, 6400, 1600 × 1 | 48 | 1896 | 2184 (1.15×) | n/a | 2146 (1.13×) | n/a |
| `mlp_fc_wgrad` | 1600, 6400, 1024 × 1 | 48 | 1875 | 2152 (1.15×) | n/a | 2124 (1.13×) | n/a |
| `mlp_proj_dgrad` | 1024, 1600, 6400 × 1 | 48 | 2021 | 2296 (1.14×) | n/a | 2269 (1.12×) | n/a |
| `mlp_proj_fwd` | 1024, 1600, 6400 × 1 | 48 | 2039 | 2315 (1.14×) | n/a | 2287 (1.12×) | n/a |
| `mlp_proj_wgrad` | 6400, 1600, 1024 × 1 | 48 | 1958 | 2238 (1.14×) | n/a | 2210 (1.13×) | n/a |
| `lm_head_dgrad` | 1024, 1600, 50257 × 1 | 1 | 15284 | 17275 (1.13×) | n/a | 17248 (1.13×) | n/a |
| `lm_head_fwd` | 1024, 50257, 1600 × 1 | 1 | 14811 | 16861 (1.14×) | n/a | 16766 (1.13×) | n/a |
| `lm_head_wgrad` | 50257, 1600, 1024 × 1 | 1 | 14386 | 16417 (1.14×) | n/a | 16294 (1.13×) | n/a |
| **per training step (ms)** | | | **1048.6** | **1321.8 (1.26×)** | **1365.8 (1.30×)** | **1240.2 (1.18×)** | **1300.8 (1.24×)** |

Production routes natively at 4 MiB: none.

Production routes natively at 16 MiB: none.

## Per row, 4096 tokens/step


**GPT-2 small (H=768, 12 heads, 12 layers), 4096 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 4 MiB | native 4 MiB | staged 16 MiB | native 16 MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 4096, 768, 768 × 1 | 48 | 462 | 562 (1.22×) | n/a | 461 (1.00×) | 468 (1.01×) |
| `qkvO_fwd` | 4096, 768, 768 × 1 | 48 | 469 | 571 (1.22×) | n/a | 469 (1.00×) | 500 (1.07×) |
| `qkvO_wgrad` | 768, 768, 4096 × 1 | 48 | 486 | 580 (1.19×) | 813 (1.67×) | 487 (1.00×) | 799 (1.64×) |
| `attn_ctx` | 4096, 64, 4096 × 12 | 24 | 2945 | 6263 (2.13×) | 5735 (1.95×) | 5888 (2.00×) | 5668 (1.92×) |
| `attn_dKV` | 4096, 64, 4096 × 12 | 24 | 2418 | 5712 (2.36×) | 4584 (1.90×) | 5354 (2.21×) | 4481 (1.85×) |
| `attn_scores` | 4096, 4096, 64 × 12 | 24 | 3409 | 6457 (1.89×) | 15416 (4.52×) | 6358 (1.86×) | 15549 (4.56×) |
| `mlp_fc_dgrad` | 4096, 3072, 768 × 1 | 12 | 1731 | 1999 (1.16×) | 1866 (1.08×) | 1916 (1.11×) | 1823 (1.05×) |
| `mlp_fc_fwd` | 4096, 3072, 768 × 1 | 12 | 1835 | 2106 (1.15×) | 2044 (1.11×) | 2023 (1.10×) | 1921 (1.05×) |
| `mlp_fc_wgrad` | 768, 3072, 4096 × 1 | 12 | 1717 | 1996 (1.16×) | 1754 (1.02×) | 1903 (1.11×) | 1718 (1.00×) |
| `mlp_proj_dgrad` | 4096, 768, 3072 × 1 | 12 | 1797 | 2081 (1.16×) | 1847 (1.03×) | 1977 (1.10×) | 1805 (1.00×) |
| `mlp_proj_fwd` | 4096, 768, 3072 × 1 | 12 | 1814 | 2101 (1.16×) | 1876 (1.03×) | 1995 (1.10×) | 1868 (1.03×) |
| `mlp_proj_wgrad` | 3072, 768, 4096 × 1 | 12 | 1773 | 2050 (1.16×) | 1754 (0.99×) | 1957 (1.10×) | 1733 (0.98×) |
| `lm_head_dgrad` | 4096, 768, 50257 × 1 | 1 | 28594 | 32429 (1.13×) | n/a | 32380 (1.13×) | n/a |
| `lm_head_fwd` | 4096, 50257, 768 × 1 | 1 | 28533 | 32131 (1.13×) | n/a | 32187 (1.13×) | n/a |
| `lm_head_wgrad` | 50257, 768, 4096 × 1 | 1 | 27048 | 30927 (1.14×) | n/a | 30632 (1.13×) | n/a |
| **per training step (ms)** | | | **490.7** | **768.1 (1.57×)** | **940.3 (1.92×)** | **726.8 (1.48×)** | **927.2 (1.89×)** |

Production routes natively at 4 MiB: `attn_ctx`, `attn_dKV`, `mlp_fc_dgrad`, `mlp_fc_fwd`, `mlp_proj_dgrad`, `mlp_proj_fwd`, `mlp_proj_wgrad`.

Production routes natively at 16 MiB: `attn_ctx`, `attn_dKV`, `mlp_fc_dgrad`, `mlp_fc_fwd`.

**GPT-2 medium (H=1024, 16 heads, 24 layers), 4096 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 4 MiB | native 4 MiB | staged 16 MiB | native 16 MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 4096, 1024, 1024 × 1 | 96 | 799 | 928 (1.16×) | n/a | 801 (1.00×) | 816 (1.02×) |
| `qkvO_fwd` | 4096, 1024, 1024 × 1 | 96 | 821 | 953 (1.16×) | n/a | 818 (1.00×) | 854 (1.04×) |
| `qkvO_wgrad` | 1024, 1024, 4096 × 1 | 96 | 806 | 934 (1.16×) | 811 (1.01×) | 802 (1.00×) | 801 (0.99×) |
| `attn_ctx` | 4096, 64, 4096 × 16 | 48 | 3987 | 8317 (2.09×) | 7589 (1.90×) | 7895 (1.98×) | 7494 (1.88×) |
| `attn_dKV` | 4096, 64, 4096 × 16 | 48 | 3201 | 7576 (2.37×) | 6068 (1.90×) | 7132 (2.23×) | 5925 (1.85×) |
| `attn_scores` | 4096, 4096, 64 × 16 | 48 | 4542 | 8604 (1.89×) | 20589 (4.53×) | 8473 (1.87×) | 20756 (4.57×) |
| `mlp_fc_dgrad` | 4096, 4096, 1024 × 1 | 24 | 3096 | 3466 (1.12×) | 3264 (1.05×) | 3341 (1.08×) | 3126 (1.01×) |
| `mlp_fc_fwd` | 4096, 4096, 1024 × 1 | 24 | 3232 | 3601 (1.11×) | 3557 (1.10×) | 3480 (1.08×) | 3229 (1.00×) |
| `mlp_fc_wgrad` | 1024, 4096, 4096 × 1 | 24 | 3098 | 3486 (1.13×) | 3105 (1.00×) | 3348 (1.08×) | 3117 (1.01×) |
| `mlp_proj_dgrad` | 4096, 1024, 4096 × 1 | 24 | 3128 | 3517 (1.12×) | 3166 (1.01×) | 3379 (1.08×) | 3133 (1.00×) |
| `mlp_proj_fwd` | 4096, 1024, 4096 × 1 | 24 | 3198 | 3593 (1.12×) | 3207 (1.00×) | 3439 (1.08×) | 3216 (1.01×) |
| `mlp_proj_wgrad` | 4096, 1024, 4096 × 1 | 24 | 3098 | 3487 (1.13×) | 3127 (1.01×) | 3348 (1.08×) | 3142 (1.01×) |
| `lm_head_dgrad` | 4096, 1024, 50257 × 1 | 1 | 38119 | 42142 (1.11×) | n/a | 42077 (1.10×) | n/a |
| `lm_head_fwd` | 4096, 50257, 1024 × 1 | 1 | 37601 | 41465 (1.10×) | n/a | 41294 (1.10×) | n/a |
| `lm_head_wgrad` | 50257, 1024, 4096 × 1 | 1 | 36319 | 40395 (1.11×) | n/a | 40084 (1.10×) | n/a |
| **per training step (ms)** | | | **1360.5** | **2077.6 (1.53×)** | **2492.4 (1.83×)** | **1971.9 (1.45×)** | **2456.2 (1.81×)** |

Production routes natively at 4 MiB: `qkvO_wgrad`, `attn_ctx`, `attn_dKV`, `mlp_fc_dgrad`, `mlp_fc_fwd`, `mlp_fc_wgrad`, `mlp_proj_dgrad`, `mlp_proj_fwd`, `mlp_proj_wgrad`.

Production routes natively at 16 MiB: `attn_ctx`, `attn_dKV`, `mlp_fc_dgrad`, `mlp_fc_fwd`.

**GPT-2 large (H=1280, 20 heads, 36 layers), 4096 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 4 MiB | native 4 MiB | staged 16 MiB | native 16 MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 4096, 1280, 1280 × 1 | 144 | 1202 | 1390 (1.16×) | 1289 (1.07×) | 1364 (1.13×) | n/a |
| `qkvO_fwd` | 4096, 1280, 1280 × 1 | 144 | 1277 | 1469 (1.15×) | 1366 (1.07×) | 1433 (1.12×) | n/a |
| `qkvO_wgrad` | 1280, 1280, 4096 × 1 | 144 | 1201 | 1389 (1.16×) | 1613 (1.34×) | 1352 (1.13×) | 1611 (1.34×) |
| `attn_ctx` | 4096, 64, 4096 × 20 | 72 | 4931 | 10414 (2.11×) | 9130 (1.85×) | 10425 (2.11×) | 9115 (1.85×) |
| `attn_dKV` | 4096, 64, 4096 × 20 | 72 | 3983 | 9456 (2.37×) | 7277 (1.83×) | 9457 (2.37×) | 7275 (1.83×) |
| `attn_scores` | 4096, 4096, 64 × 20 | 72 | 5675 | 10750 (1.89×) | 25738 (4.54×) | 10741 (1.89×) | 25706 (4.53×) |
| `mlp_fc_dgrad` | 4096, 5120, 1280 × 1 | 36 | 4690 | 5176 (1.10×) | 4846 (1.03×) | 5177 (1.10×) | 4846 (1.03×) |
| `mlp_fc_fwd` | 4096, 5120, 1280 × 1 | 36 | 4834 | 5321 (1.10×) | 5157 (1.07×) | 5321 (1.10×) | 5151 (1.07×) |
| `mlp_fc_wgrad` | 1280, 5120, 4096 × 1 | 36 | 4727 | 5224 (1.11×) | 4831 (1.02×) | 5229 (1.11×) | 4835 (1.02×) |
| `mlp_proj_dgrad` | 4096, 1280, 5120 × 1 | 36 | 4684 | 5186 (1.11×) | 4944 (1.06×) | 5184 (1.11×) | 4946 (1.06×) |
| `mlp_proj_fwd` | 4096, 1280, 5120 × 1 | 36 | 5063 | 5580 (1.10×) | 4945 (0.98×) | 5583 (1.10×) | 4944 (0.98×) |
| `mlp_proj_wgrad` | 5120, 1280, 4096 × 1 | 36 | 4729 | 5228 (1.11×) | 4833 (1.02×) | 5231 (1.11×) | 4833 (1.02×) |
| `lm_head_dgrad` | 4096, 1280, 50257 × 1 | 1 | 47643 | 51851 (1.09×) | n/a | 51852 (1.09×) | n/a |
| `lm_head_fwd` | 4096, 50257, 1280 × 1 | 1 | 47259 | 51376 (1.09×) | n/a | 51354 (1.09×) | n/a |
| `lm_head_wgrad` | 50257, 1280, 4096 × 1 | 1 | 44818 | 49101 (1.10×) | n/a | 49101 (1.10×) | n/a |
| **per training step (ms)** | | | **2754.3** | **4110.2 (1.49×)** | **4865.4 (1.77×)** | **4096.6 (1.49×)** | **4882.0 (1.77×)** |

Production routes natively at 4 MiB: `attn_ctx`, `attn_dKV`, `mlp_fc_dgrad`, `mlp_fc_fwd`, `mlp_fc_wgrad`, `mlp_proj_dgrad`, `mlp_proj_fwd`, `mlp_proj_wgrad`.

Production routes natively at 16 MiB: `attn_ctx`, `attn_dKV`, `mlp_fc_dgrad`, `mlp_fc_fwd`, `mlp_fc_wgrad`, `mlp_proj_dgrad`, `mlp_proj_fwd`, `mlp_proj_wgrad`.

**GPT-2 xl (H=1600, 25 heads, 48 layers), 4096 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 4 MiB | native 4 MiB | staged 16 MiB | native 16 MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 4096, 1600, 1600 × 1 | 192 | 1958 | 2196 (1.12×) | n/a | 2156 (1.10×) | n/a |
| `qkvO_fwd` | 4096, 1600, 1600 × 1 | 192 | 2090 | 2331 (1.12×) | n/a | 2288 (1.09×) | n/a |
| `qkvO_wgrad` | 1600, 1600, 4096 × 1 | 192 | 1887 | 2122 (1.12×) | n/a | 2083 (1.10×) | n/a |
| `attn_ctx` | 4096, 64, 4096 × 25 | 96 | 6368 | 13264 (2.08×) | 11445 (1.80×) | 13230 (2.08×) | 11433 (1.80×) |
| `attn_dKV` | 4096, 64, 4096 × 25 | 96 | 4776 | 11618 (2.43×) | 9143 (1.91×) | 11620 (2.43×) | 9142 (1.91×) |
| `attn_scores` | 4096, 4096, 64 × 25 | 96 | 7092 | 13435 (1.89×) | 32117 (4.53×) | 13423 (1.89×) | 32149 (4.53×) |
| `mlp_fc_dgrad` | 4096, 6400, 1600 × 1 | 48 | 7394 | 8028 (1.09×) | n/a | 8028 (1.09×) | n/a |
| `mlp_fc_fwd` | 4096, 6400, 1600 × 1 | 48 | 7563 | 8175 (1.08×) | n/a | 8161 (1.08×) | n/a |
| `mlp_fc_wgrad` | 1600, 6400, 4096 × 1 | 48 | 7234 | 7890 (1.09×) | n/a | 7890 (1.09×) | n/a |
| `mlp_proj_dgrad` | 4096, 1600, 6400 × 1 | 48 | 7688 | 8337 (1.08×) | n/a | 8339 (1.08×) | n/a |
| `mlp_proj_fwd` | 4096, 1600, 6400 × 1 | 48 | 8106 | 8755 (1.08×) | n/a | 8761 (1.08×) | n/a |
| `mlp_proj_wgrad` | 6400, 1600, 4096 × 1 | 48 | 7734 | 8392 (1.09×) | n/a | 8392 (1.09×) | n/a |
| `lm_head_dgrad` | 4096, 1600, 50257 × 1 | 1 | 61962 | 66402 (1.07×) | n/a | 66400 (1.07×) | n/a |
| `lm_head_fwd` | 4096, 50257, 1600 × 1 | 1 | 58587 | 62960 (1.07×) | n/a | 62732 (1.07×) | n/a |
| `lm_head_wgrad` | 50257, 1600, 4096 × 1 | 1 | 58394 | 61543 (1.05×) | n/a | 61571 (1.05×) | n/a |
| **per training step (ms)** | | | **5263.7** | **7525.6 (1.43×)** | **8906.8 (1.69×)** | **7497.4 (1.42×)** | **8884.7 (1.69×)** |

Production routes natively at 4 MiB: `attn_ctx`, `attn_dKV`.

Production routes natively at 16 MiB: `attn_ctx`, `attn_dKV`.

## First optimization round (2026-11-24)

New configurations are extension pieces: one translation unit each under
`nntile/src/kernel/gemm_chunked/ext/`, claiming exact shapes and layouts.
Five are in this commit, each measured on both engines with the benchmark
and checked against cuBLAS there and by `test_gemm_chunked`:

- `flat.cu` — all operands single-chunk: CUTLASS's SIMT engines on plain
  pointers, with a serial split-K (fixed slices, no atomics) where 48 to
  144 threadblocks left SMs idle.
- `mixed.cu` — the bridge instantiated for the flat/table operand pairs,
  with K-split schedules on the underfilled grids. The 20 MLP cases that
  the family declined before (one or two operands single-chunk) run.
- `proj.cu` — the last wave of the projection rows at 4096 tokens cut
  once along k, heads and tails in lockstep, one fixed-order fixup.
- `wgrad.cu` — 68 balanced workers over the weight-gradient tiles and a
  fixed-order reduction, instead of one threadblock per tile.
- `large1024.cu` — GPT-2 large's projection rows at 1024 tokens (80
  threadblocks) on a 5-slice split-K.

The bar of the round was the native route within 2 % of cuBLAS on each of
the 72 MLP cases (six rows, GPT-2 small, medium and large, 1024 and 4096
tokens, two engines); the next section has the count. The attention,
QKV/O and lm-head rows are untouched by the pieces. Work on the bridge's
epilogue and loaders that is not verified yet is not in this commit.

## Reading the numbers

- **Staging is the dominant cost.** Per training step it is 1.17–1.30×
  cuBLAS at 1024 tokens and 1.43–1.56× at 4096 tokens. The attention rows
  pay the most (1.8–2.4×): their per-head score matrices are whole chunks
  that must be gathered and scattered. MLP and lm-head rows pay 1.1–1.2×.
- **The native family is close to cuBLAS on the MLP rows and passes on
  half of them.** Against the round's bar it passed 7 of the 72 MLP cases
  before the round and 35 of 72 here (34 in a second run of the same
  commit: cases on the line move by a few tenths of a percent between
  runs). By piece: all-flat operands 12 of 12 (were 4); mixed layouts 16
  of 30 (were 2); projection forward and dgrad at 4096 tokens 4 of 8
  (were 1); weight gradients 3 of 8 (were 0), the other five at 1.02×;
  fc forward and dgrad at 4096 tokens 0 of 8 (1.03–1.11×); GPT-2 large at
  1024 tokens 0 of 6 (1.02–1.08×, from 1.03–1.62×). Over all rows, on the
  4 MiB engine the native route is faster than cuBLAS on 4 of the 39 rows
  it can launch at 1024 tokens and on 2 of 35 at 4096 tokens; on the
  16 MiB engine on 10 of 39 and on 4 of 37. It is faster than staging on
  the attention `ctx`/`dKV` rows of every model.
- **Mixed layouts run.** The bridge is instantiated for the flat/table
  operand pairs. On the 4 MiB class at 1024 tokens the fc rows of GPT-2
  small and medium (activations in one chunk, weights in several) run
  natively at 0.94–1.05× cuBLAS.
- **The CUTLASS flat leg on staged rows** is between 8 % faster and 3 %
  slower than the cuBLAS leg (12 rows per engine build).
- **XL is mostly out of reach.** H=1600 is neither 64 nor a 128-multiple, so
  only its attention rows are eligible.
- **Where the rest of the gap is.** Three costs, in order of size. The
  bridge's epilogue resolves the chunk table for every output element
  (about 8,400 instructions per thread against 1,130 in CUTLASS's own
  epilogue: about 7 % of a call at k = 768 and 2 % at k = 3072). Every
  8-deep k panel costs two block barriers and a stage store: a 128×128×8
  panel is 2052 clocks of pure compute for a 256-thread threadblock on
  this GPU, about 2150 in CUTLASS's SIMT engine, 2190 in the bridge on
  NN and 2300 on NT. And a grid's last wave leaves SMs idle (80 to 144
  threadblocks on 68 SMs on the worst MLP rows).

**These are scattered-chunk layouts.** In the real trainer without memory
pressure most multi-chunk tiles are contiguous chains: in a GPT-2 small
run (cap-100 cell, 4 MiB engine) 9573 of 9603 GEMM calls took the
contiguous shortcut to plain cuBLAS, 30 were staged and none ran natively.
Staging and the native family matter once chunks are scattered.

## Reproduce

```
cmake -S . -B build          # USE_CHUNKED_GEMM and USE_CUTLASS default to ON
cmake --build build --target bench_gemm_chunked_gpt2
build/nntile/tests/kernel/bench_gemm_chunked_gpt2 > routes.csv
python3 torch_nntile/tools/render_gemm_chunked_gpt2_tables.py \
    routes.csv > tables.md
```

`bench_gemm_chunked_gpt2 [class_mib,...] [tokens,...] [models,...]
[rows,...]` defaults to the engine's class, 1024 and 4096 tokens, all
four models and all 15 rows; `rows` is a list of row-name prefixes (for
example `mlp_`). The staged routes exist only at that class; a table for
another class needs an nnhaul engine built with the matching
`NNHAUL_CHUNK_SIZE_MB`. The committed files are
`gemm_chunked_gpt2_routes_4mib.csv` and `gemm_chunked_gpt2_routes_16mib.csv`
(one per engine build) and `gemm_chunked_gpt2_routes_1-2-8mib.csv`
(`--native-only`).
