# GPT-2 GEMM routes: cuBLAS vs workspace-staged vs native chunked

Baseline for the chunked-GEMM work: what each GEMM route of the nnhaul gemm
wrapper (`nntile/src/nnhaul/gemm.cc`) costs on the shapes a GPT-2 training
step issues, for the native family in `nntile/src/kernel/gemm_chunked/`.
Measured with `bench_gemm_chunked_gpt2`
(`nntile/tests/kernel/gemm_chunked_gpt2_bench.cc`) on 2026-11-23. Every
table below is the output of
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
   production routing. `n/a` when the family cannot run the row: the shape
   is outside its law (vocab 50257; H=1600), or its launchers decline a
   mixed flat/table layout.
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

| model | tokens/step | cuBLAS ms | 4 MiB: staged · native · production | 16 MiB: staged · native · production
|---|---:|---:|---|------|---
| small (H=768, 12 heads, 12 layers) | 1024 | 92.4 | 115.9 (1.25×) · 132.1 (1.43×) · 115.9 (1.25×) | 109.9 (1.19×) · 131.3 (1.42×) · 109.9 (1.19×) |
| small (H=768, 12 heads, 12 layers) | 4096 | 491.9 | 767.0 (1.56×) · 956.5 (1.94×) · 720.1 (1.46×) | 728.2 (1.48×) · 937.4 (1.91×) · 699.2 (1.42×) |
| medium (H=1024, 16 heads, 24 layers) | 1024 | 247.1 | 305.9 (1.24×) · 325.6 (1.32×) · 299.3 (1.21×) | 288.2 (1.17×) · 315.7 (1.28×) · 288.2 (1.17×) |
| medium (H=1024, 16 heads, 24 layers) | 4096 | 1361.0 | 2078.6 (1.53×) · 2502.1 (1.84×) · 1927.5 (1.42×) | 1972.4 (1.45×) · 2487.4 (1.83×) · 1892.8 (1.39×) |
| large (H=1280, 20 heads, 36 layers) | 1024 | 524.5 | 679.1 (1.30×) · 766.4 (1.46×) · 660.2 (1.26×) | 635.9 (1.21×) · 779.0 (1.48×) · 634.4 (1.21×) |
| large (H=1280, 20 heads, 36 layers) | 4096 | 2756.5 | 4112.5 (1.49×) · 4880.8 (1.77×) · 3797.3 (1.38×) | 4099.4 (1.49×) · 4901.5 (1.78×) · 3787.0 (1.38×) |
| xl (H=1600, 25 heads, 48 layers) | 1024 | 1050.7 | 1323.7 (1.26×) · 1368.3 (1.30×) · 1323.7 (1.26×) | 1240.6 (1.18×) · 1300.9 (1.24×) · 1240.9 (1.18×) |
| xl (H=1600, 25 heads, 48 layers) | 4096 | 5266.4 | 7524.7 (1.43×) · 8906.7 (1.69×) · 7112.4 (1.35×) | 7503.7 (1.42×) · 8890.2 (1.69×) · 7089.9 (1.35×) |

## Native family alone at the other classes

| model | tokens/step | cuBLAS ms | 1 MiB class: native, rows launched | 2 MiB class: native, rows launched | 8 MiB class: native, rows launched
|---|---:|---:|---|------|------|---
| small | 1024 | 92.4 | 129.4 (1.40×), 12/15 | 129.4 (1.40×), 12/15 | 124.0 (1.34×), 8/15 |
| small | 4096 | 491.4 | 938.3 (1.91×), 12/15 | 938.2 (1.91×), 12/15 | 935.8 (1.91×), 10/15 |
| medium | 1024 | 247.3 | 317.3 (1.28×), 12/15 | 316.6 (1.28×), 12/15 | 308.7 (1.25×), 8/15 |
| medium | 4096 | 1360.0 | 2474.7 (1.82×), 12/15 | 2475.2 (1.82×), 12/15 | 2466.0 (1.81×), 10/15 |
| large | 1024 | 523.1 | 760.7 (1.45×), 12/15 | 759.4 (1.46×), 12/15 | 742.9 (1.42×), 8/15 |
| large | 4096 | 2758.7 | 4864.7 (1.76×), 12/15 | 4867.3 (1.77×), 12/15 | 4843.1 (1.76×), 10/15 |
| xl | 1024 | 1052.5 | 1229.7 (1.17×), 3/15 | 1225.3 (1.17×), 3/15 | 1227.3 (1.17×), 3/15 |
| xl | 4096 | 5273.8 | 8576.3 (1.63×), 3/15 | 8575.2 (1.63×), 3/15 | 8576.8 (1.63×), 3/15 |

## Per row, 1024 tokens/step


**GPT-2 small (H=768, 12 heads, 12 layers), 1024 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 4 MiB | native 4 MiB | staged 16 MiB | native 16 MiB |
|---|---|---:|---:|---|------|------|------|---
| `qkvO_dgrad` | 1024, 768, 768 × 1 | 48 | 158 | 158 (1.00×) | 171 (1.09×) | 158 (1.00×) | 171 (1.08×) |
| `qkvO_fwd` | 1024, 768, 768 × 1 | 48 | 136 | 136 (1.00×) | 174 (1.28×) | 136 (1.00×) | 174 (1.28×) |
| `qkvO_wgrad` | 768, 768, 1024 × 1 | 48 | 136 | 136 (1.01×) | 217 (1.60×) | 136 (1.00×) | 217 (1.60×) |
| `attn_ctx` | 1024, 64, 1024 × 12 | 24 | 234 | 418 (1.79×) | 385 (1.65×) | 419 (1.79×) | 385 (1.65×) |
| `attn_dKV` | 1024, 64, 1024 × 12 | 24 | 217 | 402 (1.85×) | 309 (1.42×) | 403 (1.86×) | 309 (1.43×) |
| `attn_scores` | 1024, 1024, 64 × 12 | 24 | 225 | 414 (1.84×) | 890 (3.96×) | 414 (1.84×) | 891 (3.96×) |
| `mlp_fc_dgrad` | 1024, 3072, 768 × 1 | 12 | 463 | 547 (1.18×) | n/a | 463 (1.00×) | 471 (1.02×) |
| `mlp_fc_fwd` | 1024, 3072, 768 × 1 | 12 | 471 | 558 (1.18×) | n/a | 473 (1.00×) | 490 (1.04×) |
| `mlp_fc_wgrad` | 768, 3072, 1024 × 1 | 12 | 507 | 588 (1.16×) | n/a | 506 (1.00×) | 604 (1.19×) |
| `mlp_proj_dgrad` | 1024, 768, 3072 × 1 | 12 | 465 | 548 (1.18×) | 621 (1.34×) | 465 (1.00×) | 616 (1.33×) |
| `mlp_proj_fwd` | 1024, 768, 3072 × 1 | 12 | 495 | 578 (1.17×) | 627 (1.26×) | 496 (1.00×) | 644 (1.30×) |
| `mlp_proj_wgrad` | 3072, 768, 1024 × 1 | 12 | 460 | 547 (1.19×) | n/a | 461 (1.00×) | 604 (1.31×) |
| `lm_head_dgrad` | 1024, 768, 50257 × 1 | 1 | 7182 | 8518 (1.19×) | n/a | 8518 (1.19×) | n/a |
| `lm_head_fwd` | 1024, 50257, 768 × 1 | 1 | 7219 | 8567 (1.19×) | n/a | 8516 (1.18×) | n/a |
| `lm_head_wgrad` | 50257, 768, 1024 × 1 | 1 | 6864 | 8188 (1.19×) | n/a | 8188 (1.19×) | n/a |
| **per training step (ms)** | | | **92.4** | **115.9 (1.25×)** | **132.1 (1.43×)** | **109.9 (1.19×)** | **131.3 (1.42×)** |

Production routes natively at 4 MiB: none.

Production routes natively at 16 MiB: none.

**GPT-2 medium (H=1024, 16 heads, 24 layers), 1024 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 4 MiB | native 4 MiB | staged 16 MiB | native 16 MiB |
|---|---|---:|---:|---|------|------|------|---
| `qkvO_dgrad` | 1024, 1024, 1024 × 1 | 96 | 219 | 218 (1.00×) | 221 (1.01×) | 219 (1.00×) | 221 (1.01×) |
| `qkvO_fwd` | 1024, 1024, 1024 × 1 | 96 | 235 | 235 (1.00×) | 225 (0.96×) | 236 (1.00×) | 225 (0.96×) |
| `qkvO_wgrad` | 1024, 1024, 1024 × 1 | 96 | 215 | 215 (1.00×) | 218 (1.01×) | 216 (1.00×) | 218 (1.01×) |
| `attn_ctx` | 1024, 64, 1024 × 16 | 48 | 256 | 508 (1.99×) | 499 (1.95×) | 510 (1.99×) | 499 (1.95×) |
| `attn_dKV` | 1024, 64, 1024 × 16 | 48 | 222 | 471 (2.13×) | 400 (1.80×) | 473 (2.14×) | 400 (1.80×) |
| `attn_scores` | 1024, 1024, 64 × 16 | 48 | 295 | 546 (1.85×) | 1178 (3.99×) | 547 (1.85×) | 1178 (3.99×) |
| `mlp_fc_dgrad` | 1024, 4096, 1024 × 1 | 24 | 797 | 926 (1.16×) | n/a | 797 (1.00×) | 825 (1.03×) |
| `mlp_fc_fwd` | 1024, 4096, 1024 × 1 | 24 | 824 | 952 (1.16×) | n/a | 824 (1.00×) | 855 (1.04×) |
| `mlp_fc_wgrad` | 1024, 4096, 1024 × 1 | 24 | 788 | 916 (1.16×) | n/a | 788 (1.00×) | 811 (1.03×) |
| `mlp_proj_dgrad` | 1024, 1024, 4096 × 1 | 24 | 817 | 940 (1.15×) | 827 (1.01×) | 819 (1.00×) | 825 (1.01×) |
| `mlp_proj_fwd` | 1024, 1024, 4096 × 1 | 24 | 883 | 1006 (1.14×) | 857 (0.97×) | 885 (1.00×) | 854 (0.97×) |
| `mlp_proj_wgrad` | 4096, 1024, 1024 × 1 | 24 | 788 | 918 (1.16×) | n/a | 788 (1.00×) | 811 (1.03×) |
| `lm_head_dgrad` | 1024, 1024, 50257 × 1 | 1 | 9582 | 11088 (1.16×) | n/a | 11090 (1.16×) | n/a |
| `lm_head_fwd` | 1024, 50257, 1024 × 1 | 1 | 9506 | 11008 (1.16×) | n/a | 11004 (1.16×) | n/a |
| `lm_head_wgrad` | 50257, 1024, 1024 × 1 | 1 | 9115 | 10627 (1.17×) | n/a | 10629 (1.17×) | n/a |
| **per training step (ms)** | | | **247.1** | **305.9 (1.24×)** | **325.6 (1.32×)** | **288.2 (1.17×)** | **315.7 (1.28×)** |

Production routes natively at 4 MiB: `mlp_proj_dgrad`, `mlp_proj_fwd`.

Production routes natively at 16 MiB: none.

**GPT-2 large (H=1280, 20 heads, 36 layers), 1024 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 4 MiB | native 4 MiB | staged 16 MiB | native 16 MiB |
|---|---|---:|---:|---|------|------|------|---
| `qkvO_dgrad` | 1024, 1280, 1280 × 1 | 144 | 328 | 395 (1.21×) | 523 (1.60×) | 327 (1.00×) | 522 (1.59×) |
| `qkvO_fwd` | 1024, 1280, 1280 × 1 | 144 | 353 | 418 (1.18×) | 545 (1.54×) | 353 (1.00×) | 543 (1.54×) |
| `qkvO_wgrad` | 1280, 1280, 1024 × 1 | 144 | 317 | 385 (1.21×) | 434 (1.37×) | 319 (1.01×) | 413 (1.30×) |
| `attn_ctx` | 1024, 64, 1024 × 20 | 72 | 357 | 720 (2.01×) | 633 (1.77×) | 661 (1.85×) | 617 (1.73×) |
| `attn_dKV` | 1024, 64, 1024 × 20 | 72 | 317 | 686 (2.16×) | 504 (1.59×) | 627 (1.97×) | 494 (1.56×) |
| `attn_scores` | 1024, 1024, 64 × 20 | 72 | 367 | 719 (1.96×) | 1470 (4.01×) | 680 (1.85×) | 1469 (4.01×) |
| `mlp_fc_dgrad` | 1024, 5120, 1280 × 1 | 36 | 1239 | 1442 (1.16×) | 1276 (1.03×) | 1419 (1.14×) | n/a |
| `mlp_fc_fwd` | 1024, 5120, 1280 × 1 | 36 | 1276 | 1480 (1.16×) | 1352 (1.06×) | 1466 (1.15×) | n/a |
| `mlp_fc_wgrad` | 1280, 5120, 1024 × 1 | 36 | 1206 | 1399 (1.16×) | 1294 (1.07×) | 1393 (1.16×) | n/a |
| `mlp_proj_dgrad` | 1024, 1280, 5120 × 1 | 36 | 1254 | 1443 (1.15×) | 2023 (1.61×) | 1424 (1.14×) | 2042 (1.63×) |
| `mlp_proj_fwd` | 1024, 1280, 5120 × 1 | 36 | 1345 | 1540 (1.15×) | 1716 (1.28×) | 1521 (1.13×) | 1741 (1.29×) |
| `mlp_proj_wgrad` | 5120, 1280, 1024 × 1 | 36 | 1206 | 1399 (1.16×) | 1290 (1.07×) | 1394 (1.16×) | n/a |
| `lm_head_dgrad` | 1024, 1280, 50257 × 1 | 1 | 11497 | 13228 (1.15×) | n/a | 13210 (1.15×) | n/a |
| `lm_head_fwd` | 1024, 50257, 1280 × 1 | 1 | 11944 | 13771 (1.15×) | n/a | 13647 (1.14×) | n/a |
| `lm_head_wgrad` | 50257, 1280, 1024 × 1 | 1 | 11423 | 13162 (1.15×) | n/a | 13161 (1.15×) | n/a |
| **per training step (ms)** | | | **524.5** | **679.1 (1.29×)** | **766.4 (1.46×)** | **635.9 (1.21×)** | **779.0 (1.49×)** |

Production routes natively at 4 MiB: `mlp_fc_dgrad`, `mlp_fc_fwd`, `mlp_fc_wgrad`, `mlp_proj_wgrad`.

Production routes natively at 16 MiB: none.

**GPT-2 xl (H=1600, 25 heads, 48 layers), 1024 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 4 MiB | native 4 MiB | staged 16 MiB | native 16 MiB |
|---|---|---:|---:|---|------|------|------|---
| `qkvO_dgrad` | 1024, 1600, 1600 × 1 | 192 | 517 | 607 (1.17×) | n/a | 516 (1.00×) | n/a |
| `qkvO_fwd` | 1024, 1600, 1600 × 1 | 192 | 632 | 722 (1.14×) | n/a | 630 (1.00×) | n/a |
| `qkvO_wgrad` | 1600, 1600, 1024 × 1 | 192 | 501 | 594 (1.19×) | n/a | 500 (1.00×) | n/a |
| `attn_ctx` | 1024, 64, 1024 × 25 | 96 | 469 | 957 (2.04×) | 758 (1.62×) | 889 (1.89×) | 735 (1.57×) |
| `attn_dKV` | 1024, 64, 1024 × 25 | 96 | 414 | 876 (2.11×) | 602 (1.45×) | 797 (1.92×) | 588 (1.42×) |
| `attn_scores` | 1024, 1024, 64 × 25 | 96 | 455 | 895 (1.97×) | 1834 (4.03×) | 844 (1.85×) | 1833 (4.03×) |
| `mlp_fc_dgrad` | 1024, 6400, 1600 × 1 | 48 | 1873 | 2164 (1.16×) | n/a | 2119 (1.13×) | n/a |
| `mlp_fc_fwd` | 1024, 6400, 1600 × 1 | 48 | 1905 | 2189 (1.15×) | n/a | 2149 (1.13×) | n/a |
| `mlp_fc_wgrad` | 1600, 6400, 1024 × 1 | 48 | 1875 | 2152 (1.15×) | n/a | 2124 (1.13×) | n/a |
| `mlp_proj_dgrad` | 1024, 1600, 6400 × 1 | 48 | 2034 | 2319 (1.14×) | n/a | 2267 (1.11×) | n/a |
| `mlp_proj_fwd` | 1024, 1600, 6400 × 1 | 48 | 2041 | 2316 (1.13×) | n/a | 2286 (1.12×) | n/a |
| `mlp_proj_wgrad` | 6400, 1600, 1024 × 1 | 48 | 1959 | 2238 (1.14×) | n/a | 2210 (1.13×) | n/a |
| `lm_head_dgrad` | 1024, 1600, 50257 × 1 | 1 | 15284 | 17275 (1.13×) | n/a | 17250 (1.13×) | n/a |
| `lm_head_fwd` | 1024, 50257, 1600 × 1 | 1 | 14799 | 16865 (1.14×) | n/a | 16738 (1.13×) | n/a |
| `lm_head_wgrad` | 50257, 1600, 1024 × 1 | 1 | 14403 | 16419 (1.14×) | n/a | 16333 (1.13×) | n/a |
| **per training step (ms)** | | | **1050.7** | **1323.7 (1.26×)** | **1368.3 (1.30×)** | **1240.6 (1.18×)** | **1300.9 (1.24×)** |

Production routes natively at 4 MiB: none.

Production routes natively at 16 MiB: none.

## Per row, 4096 tokens/step


**GPT-2 small (H=768, 12 heads, 12 layers), 4096 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 4 MiB | native 4 MiB | staged 16 MiB | native 16 MiB |
|---|---|---:|---:|---|------|------|------|---
| `qkvO_dgrad` | 4096, 768, 768 × 1 | 48 | 463 | 558 (1.21×) | n/a | 463 (1.00×) | 470 (1.02×) |
| `qkvO_fwd` | 4096, 768, 768 × 1 | 48 | 472 | 569 (1.21×) | n/a | 469 (0.99×) | 500 (1.06×) |
| `qkvO_wgrad` | 768, 768, 4096 × 1 | 48 | 486 | 580 (1.19×) | 815 (1.68×) | 487 (1.00×) | 799 (1.64×) |
| `attn_ctx` | 4096, 64, 4096 × 12 | 24 | 2974 | 6230 (2.09×) | 5736 (1.93×) | 5921 (1.99×) | 5667 (1.91×) |
| `attn_dKV` | 4096, 64, 4096 × 12 | 24 | 2427 | 5716 (2.36×) | 4585 (1.89×) | 5367 (2.21×) | 4480 (1.85×) |
| `attn_scores` | 4096, 4096, 64 × 12 | 24 | 3412 | 6459 (1.89×) | 15428 (4.52×) | 6361 (1.86×) | 15560 (4.56×) |
| `mlp_fc_dgrad` | 4096, 3072, 768 × 1 | 12 | 1730 | 2000 (1.16×) | 1866 (1.08×) | 1916 (1.11×) | 1858 (1.07×) |
| `mlp_fc_fwd` | 4096, 3072, 768 × 1 | 12 | 1835 | 2105 (1.15×) | 2047 (1.12×) | 2023 (1.10×) | 1998 (1.09×) |
| `mlp_fc_wgrad` | 768, 3072, 4096 × 1 | 12 | 1717 | 1995 (1.16×) | 2412 (1.40×) | 1903 (1.11×) | n/a |
| `mlp_proj_dgrad` | 4096, 768, 3072 × 1 | 12 | 1793 | 2065 (1.15×) | 1852 (1.03×) | 1985 (1.11×) | n/a |
| `mlp_proj_fwd` | 4096, 768, 3072 × 1 | 12 | 1820 | 2100 (1.15×) | 1881 (1.03×) | 2006 (1.10×) | n/a |
| `mlp_proj_wgrad` | 3072, 768, 4096 × 1 | 12 | 1773 | 2050 (1.16×) | 2416 (1.36×) | 1956 (1.10×) | n/a |
| `lm_head_dgrad` | 4096, 768, 50257 × 1 | 1 | 28593 | 32428 (1.13×) | n/a | 32381 (1.13×) | n/a |
| `lm_head_fwd` | 4096, 50257, 768 × 1 | 1 | 28489 | 32191 (1.13×) | n/a | 32022 (1.12×) | n/a |
| `lm_head_wgrad` | 50257, 768, 4096 × 1 | 1 | 27049 | 30927 (1.14×) | n/a | 30632 (1.13×) | n/a |
| **per training step (ms)** | | | **491.9** | **767.0 (1.56×)** | **956.5 (1.94×)** | **728.2 (1.48×)** | **937.4 (1.91×)** |

Production routes natively at 4 MiB: `attn_ctx`, `attn_dKV`, `mlp_fc_dgrad`, `mlp_fc_fwd`, `mlp_proj_dgrad`, `mlp_proj_fwd`, `mlp_proj_wgrad`.

Production routes natively at 16 MiB: `attn_ctx`, `attn_dKV`, `mlp_fc_dgrad`, `mlp_fc_fwd`.

**GPT-2 medium (H=1024, 16 heads, 24 layers), 4096 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 4 MiB | native 4 MiB | staged 16 MiB | native 16 MiB |
|---|---|---:|---:|---|------|------|------|---
| `qkvO_dgrad` | 4096, 1024, 1024 × 1 | 96 | 798 | 928 (1.16×) | n/a | 798 (1.00×) | 821 (1.03×) |
| `qkvO_fwd` | 4096, 1024, 1024 × 1 | 96 | 824 | 954 (1.16×) | n/a | 821 (1.00×) | 855 (1.04×) |
| `qkvO_wgrad` | 1024, 1024, 4096 × 1 | 96 | 808 | 935 (1.16×) | 813 (1.01×) | 808 (1.00×) | 805 (1.00×) |
| `attn_ctx` | 4096, 64, 4096 × 16 | 48 | 3989 | 8330 (2.09×) | 7587 (1.90×) | 7885 (1.98×) | 7495 (1.88×) |
| `attn_dKV` | 4096, 64, 4096 × 16 | 48 | 3195 | 7576 (2.37×) | 6066 (1.90×) | 7125 (2.23×) | 5927 (1.86×) |
| `attn_scores` | 4096, 4096, 64 × 16 | 48 | 4546 | 8606 (1.89×) | 20578 (4.53×) | 8479 (1.87×) | 20762 (4.57×) |
| `mlp_fc_dgrad` | 4096, 4096, 1024 × 1 | 24 | 3097 | 3467 (1.12×) | 3264 (1.05×) | 3342 (1.08×) | 3243 (1.05×) |
| `mlp_fc_fwd` | 4096, 4096, 1024 × 1 | 24 | 3230 | 3602 (1.12×) | 3555 (1.10×) | 3475 (1.08×) | 3426 (1.06×) |
| `mlp_fc_wgrad` | 1024, 4096, 4096 × 1 | 24 | 3098 | 3486 (1.13×) | 3209 (1.04×) | 3348 (1.08×) | n/a |
| `mlp_proj_dgrad` | 4096, 1024, 4096 × 1 | 24 | 3128 | 3517 (1.12×) | 3207 (1.03×) | 3379 (1.08×) | n/a |
| `mlp_proj_fwd` | 4096, 1024, 4096 × 1 | 24 | 3209 | 3592 (1.12×) | 3366 (1.05×) | 3460 (1.08×) | n/a |
| `mlp_proj_wgrad` | 4096, 1024, 4096 × 1 | 24 | 3098 | 3487 (1.13×) | 3245 (1.05×) | 3348 (1.08×) | n/a |
| `lm_head_dgrad` | 4096, 1024, 50257 × 1 | 1 | 38118 | 42141 (1.11×) | n/a | 42077 (1.10×) | n/a |
| `lm_head_fwd` | 4096, 50257, 1024 × 1 | 1 | 37623 | 41454 (1.10×) | n/a | 41340 (1.10×) | n/a |
| `lm_head_wgrad` | 50257, 1024, 4096 × 1 | 1 | 36320 | 40395 (1.11×) | n/a | 40086 (1.10×) | n/a |
| **per training step (ms)** | | | **1361.0** | **2078.6 (1.53×)** | **2502.1 (1.84×)** | **1972.4 (1.45×)** | **2487.4 (1.83×)** |

Production routes natively at 4 MiB: `qkvO_wgrad`, `attn_ctx`, `attn_dKV`, `mlp_fc_dgrad`, `mlp_fc_fwd`, `mlp_fc_wgrad`, `mlp_proj_dgrad`, `mlp_proj_fwd`, `mlp_proj_wgrad`.

Production routes natively at 16 MiB: `attn_ctx`, `attn_dKV`, `mlp_fc_dgrad`, `mlp_fc_fwd`.

**GPT-2 large (H=1280, 20 heads, 36 layers), 4096 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 4 MiB | native 4 MiB | staged 16 MiB | native 16 MiB |
|---|---|---:|---:|---|------|------|------|---
| `qkvO_dgrad` | 4096, 1280, 1280 × 1 | 144 | 1202 | 1390 (1.16×) | 1290 (1.07×) | 1364 (1.13×) | n/a |
| `qkvO_fwd` | 4096, 1280, 1280 × 1 | 144 | 1282 | 1469 (1.15×) | 1366 (1.07×) | 1439 (1.12×) | n/a |
| `qkvO_wgrad` | 1280, 1280, 4096 × 1 | 144 | 1199 | 1377 (1.15×) | 1614 (1.35×) | 1361 (1.13×) | 1610 (1.34×) |
| `attn_ctx` | 4096, 64, 4096 × 20 | 72 | 4951 | 10455 (2.11×) | 9135 (1.84×) | 10451 (2.11×) | 9128 (1.84×) |
| `attn_dKV` | 4096, 64, 4096 × 20 | 72 | 3983 | 9456 (2.37×) | 7277 (1.83×) | 9456 (2.37×) | 7276 (1.83×) |
| `attn_scores` | 4096, 4096, 64 × 20 | 72 | 5680 | 10752 (1.89×) | 25731 (4.53×) | 10750 (1.89×) | 25732 (4.53×) |
| `mlp_fc_dgrad` | 4096, 5120, 1280 × 1 | 36 | 4690 | 5176 (1.10×) | 4847 (1.03×) | 5176 (1.10×) | 4844 (1.03×) |
| `mlp_fc_fwd` | 4096, 5120, 1280 × 1 | 36 | 4837 | 5328 (1.10×) | 5161 (1.07×) | 5319 (1.10×) | 5147 (1.06×) |
| `mlp_fc_wgrad` | 1280, 5120, 4096 × 1 | 36 | 4724 | 5228 (1.11×) | 4880 (1.03×) | 5216 (1.10×) | 4857 (1.03×) |
| `mlp_proj_dgrad` | 4096, 1280, 5120 × 1 | 36 | 4684 | 5185 (1.11×) | 5039 (1.08×) | 5186 (1.11×) | 5101 (1.09×) |
| `mlp_proj_fwd` | 4096, 1280, 5120 × 1 | 36 | 5065 | 5588 (1.10×) | 5146 (1.02×) | 5551 (1.10×) | 5154 (1.02×) |
| `mlp_proj_wgrad` | 5120, 1280, 4096 × 1 | 36 | 4725 | 5230 (1.11×) | 4907 (1.04×) | 5221 (1.10×) | 4893 (1.04×) |
| `lm_head_dgrad` | 4096, 1280, 50257 × 1 | 1 | 47643 | 51853 (1.09×) | n/a | 51852 (1.09×) | n/a |
| `lm_head_fwd` | 4096, 50257, 1280 × 1 | 1 | 47267 | 51381 (1.09×) | n/a | 51351 (1.09×) | n/a |
| `lm_head_wgrad` | 50257, 1280, 4096 × 1 | 1 | 44833 | 49101 (1.10×) | n/a | 49128 (1.10×) | n/a |
| **per training step (ms)** | | | **2756.5** | **4112.5 (1.49×)** | **4880.8 (1.77×)** | **4099.4 (1.49×)** | **4901.5 (1.78×)** |

Production routes natively at 4 MiB: `attn_ctx`, `attn_dKV`, `mlp_fc_dgrad`, `mlp_fc_fwd`, `mlp_fc_wgrad`, `mlp_proj_dgrad`, `mlp_proj_fwd`, `mlp_proj_wgrad`.

Production routes natively at 16 MiB: `attn_ctx`, `attn_dKV`, `mlp_fc_dgrad`, `mlp_fc_fwd`, `mlp_fc_wgrad`, `mlp_proj_dgrad`, `mlp_proj_fwd`, `mlp_proj_wgrad`.

**GPT-2 xl (H=1600, 25 heads, 48 layers), 4096 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 4 MiB | native 4 MiB | staged 16 MiB | native 16 MiB |
|---|---|---:|---:|---|------|------|------|---
| `qkvO_dgrad` | 4096, 1600, 1600 × 1 | 192 | 1958 | 2196 (1.12×) | n/a | 2155 (1.10×) | n/a |
| `qkvO_fwd` | 4096, 1600, 1600 × 1 | 192 | 2093 | 2325 (1.11×) | n/a | 2292 (1.09×) | n/a |
| `qkvO_wgrad` | 1600, 1600, 4096 × 1 | 192 | 1884 | 2123 (1.13×) | n/a | 2085 (1.11×) | n/a |
| `attn_ctx` | 4096, 64, 4096 × 25 | 96 | 6403 | 13268 (2.07×) | 11450 (1.79×) | 13278 (2.07×) | 11445 (1.79×) |
| `attn_dKV` | 4096, 64, 4096 × 25 | 96 | 4776 | 11618 (2.43×) | 9142 (1.91×) | 11619 (2.43×) | 9142 (1.91×) |
| `attn_scores` | 4096, 4096, 64 × 25 | 96 | 7092 | 13423 (1.89×) | 32113 (4.53×) | 13436 (1.89×) | 32188 (4.54×) |
| `mlp_fc_dgrad` | 4096, 6400, 1600 × 1 | 48 | 7395 | 8029 (1.09×) | n/a | 8028 (1.09×) | n/a |
| `mlp_fc_fwd` | 4096, 6400, 1600 × 1 | 48 | 7565 | 8177 (1.08×) | n/a | 8161 (1.08×) | n/a |
| `mlp_fc_wgrad` | 1600, 6400, 4096 × 1 | 48 | 7234 | 7890 (1.09×) | n/a | 7889 (1.09×) | n/a |
| `mlp_proj_dgrad` | 4096, 1600, 6400 × 1 | 48 | 7688 | 8338 (1.08×) | n/a | 8338 (1.08×) | n/a |
| `mlp_proj_fwd` | 4096, 1600, 6400 × 1 | 48 | 8092 | 8769 (1.08×) | n/a | 8751 (1.08×) | n/a |
| `mlp_proj_wgrad` | 6400, 1600, 4096 × 1 | 48 | 7734 | 8392 (1.09×) | n/a | 8391 (1.08×) | n/a |
| `lm_head_dgrad` | 4096, 1600, 50257 × 1 | 1 | 61962 | 66400 (1.07×) | n/a | 66402 (1.07×) | n/a |
| `lm_head_fwd` | 4096, 50257, 1600 × 1 | 1 | 58576 | 62942 (1.07×) | n/a | 62933 (1.07×) | n/a |
| `lm_head_wgrad` | 50257, 1600, 4096 × 1 | 1 | 58397 | 61590 (1.05×) | n/a | 61571 (1.05×) | n/a |
| **per training step (ms)** | | | **5266.4** | **7524.7 (1.43×)** | **8906.7 (1.69×)** | **7503.7 (1.42×)** | **8890.2 (1.69×)** |

Production routes natively at 4 MiB: `attn_ctx`, `attn_dKV`.

Production routes natively at 16 MiB: `attn_ctx`, `attn_dKV`.

## Reading the numbers

- **Staging is the dominant cost.** Per training step it is 1.17–1.30×
  cuBLAS at 1024 tokens and 1.42–1.56× at 4096 tokens. The attention rows
  pay the most (1.8–2.4×): their per-head score matrices are whole chunks
  that must be gathered and scattered. MLP and lm-head rows pay 1.1–1.2×.
- **The native family loses to cuBLAS almost everywhere.** On the 4 MiB
  engine it is faster than cuBLAS on 2 of the 31 rows it can launch at
  1024 tokens (medium `qkvO_fwd` 0.96× and `mlp_proj_fwd` 0.97×) and on
  none of the 45 at 4096 tokens. It is faster than staging on the attention
  `ctx`/`dKV` rows of every model, and on most of the MLP rows it can
  launch at 4096 tokens.
- **Mixed layouts are declined.** The CUTLASS bridge only instantiates
  matched operand modes, so a GEMM whose operands mix single-chunk and
  multi-chunk is `n/a`. At the 4 MiB class and 1024 tokens that is the
  layout of four of the six MLP GEMMs of GPT-2 small and medium.
- **The CUTLASS flat leg on staged rows** is between 8 % faster and 3 %
  slower than the cuBLAS leg (12 rows per engine build).
- **XL is mostly out of reach.** H=1600 is neither 64 nor a 128-multiple, so
  only its attention rows are eligible.

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

`bench_gemm_chunked_gpt2 [class_mib,...] [tokens,...] [models,...]` defaults
to the engine's class, 1024 and 4096 tokens, and all four models. The
staged routes exist only at that class; a table for another class needs an
nnhaul engine built with the matching `NNHAUL_CHUNK_SIZE_MB`. The committed
files are `gemm_chunked_gpt2_routes_4mib.csv` and
`gemm_chunked_gpt2_routes_16mib.csv` (one per engine build) and
`gemm_chunked_gpt2_routes_1-2-8mib.csv` (`--native-only`).
