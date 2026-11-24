# GEMM routes by chunk class: 1, 2 and 4 MiB (GPT-2 S to XL)

What the three GEMM routes cost on every GEMM of a GPT-2 training step
when the engine's chunk class is 1, 2 or 4 MiB: contiguous cuBLAS, the
workspace-staged route with its copies, and the native chunked family of
`nntile/src/kernel/gemm_chunked/`. Measured with `bench_gemm_chunked_gpt2`
(`nntile/tests/kernel/gemm_chunked_gpt2_bench.cc`) on 2026-11-24 at commit
`12aa3564` of branch `gemm-chunked`. Every table below is the output of
`torch_nntile/tools/render_chunk_class_tables.py` on the three CSV files
committed next to this doc.

**Routes.**

1. **cuBLAS** — contiguous: plain `kernel::cublas::gemm` on flat
   operands. No chunk table and nothing to copy; the reference.
2. **staged** — the wrapper's workspace-staged body: every multi-chunk
   operand is gathered into a contiguous workspace slot
   (`copy::chunked_segments`), cuBLAS runs on the slots, and a multi-chunk
   C is scattered back (`copy::chunked`). The copies are inside the timed
   region. A single-chunk operand is not staged. Bitwise equal to cuBLAS.
3. **chunked** — the native family in place on the chunk tables
   (`launch_forced`: its best configuration for the shape, whatever the
   production ownership policy says). No workspace. `n/a` where the family
   cannot run the row.

**Rows.** The 15 distinct GEMM shapes of one training step, forward and
backward, for GPT-2 small (H=768), medium (H=1024), large (H=1280) and XL
(H=1600), at 1024 and 4096 tokens per step (batch × seq): Q/K/V and O
projections (`qkvO_*`, four calls per layer), the three batched attention
shapes (two calls per layer each), the six MLP shapes and the three
lm-head shapes.

**Chunk class and layout.** A multi-chunk operand is a table of
whole-class chunks with a shorter last one; an operand that fits one
chunk is a single flat allocation. The class therefore decides which
operands are chunked at all (the table after the first one). Step totals
are sums of calls × median: GEMM time only, with the chunked total taking
the native route where the family runs the row and staged elsewhere.

**Conditions.** NVIDIA GeForce RTX 2080 Ti (sm_75), CUDA 12.4, cuBLAS
12.4.5.8, CUTLASS 2.11.0; one process on the GPU, SM clock pinned at
1350 MHz. Median of at least 50 timed iterations (CUDA event pairs, five
interleaved rounds). Every route was checked against the cuBLAS result on
the same data before timing: staged is bitwise identical on all 120 rows
of every class, the chunked route is within 3.6e-6 relative.

Three engine builds were measured, because the staged body's copy kernels
are compiled for one chunk class: for each class an nnhaul engine built
with that `NNHAUL_CHUNK_SIZE_MB` and an nntile build against it. The
native family takes the class at runtime. cuBLAS on flat operands does
not depend on the class; it was measured three times, once per build, and
repeated within 1 % on 100 of the 120 rows (largest spread 4.4 %): the
tables show the mean, so a per-row difference of a few percent between
classes is inside the run-to-run variation.

## GEMM time per training step

| model | tokens/step | cuBLAS ms | 1 MiB class: staged · chunked | 2 MiB class: staged · chunked | 4 MiB class: staged · chunked |
|---|---:|---:|---|---|---|
| small (H=768, 12 heads, 12 layers) | 1024 | 92.4 | 124.3 (1.34×) · 133.7 (1.44×) | 124.0 (1.34×) · 133.7 (1.45×) | 115.7 (1.25×) · 125.1 (1.36×) |
| small (H=768, 12 heads, 12 layers) | 4096 | 491.4 | 769.1 (1.57×) · 934.9 (1.90×) | 768.5 (1.56×) · 935.5 (1.90×) | 767.5 (1.56×) · 939.8 (1.91×) |
| medium (H=1024, 16 heads, 24 layers) | 1024 | 246.7 | 328.9 (1.33×) · 322.0 (1.30×) | 328.4 (1.33×) · 322.0 (1.30×) | 305.1 (1.24×) · 314.2 (1.28×) |
| medium (H=1024, 16 heads, 24 layers) | 4096 | 1360.1 | 2081.7 (1.53×) · 2477.5 (1.82×) | 2083.2 (1.53×) · 2479.9 (1.82×) | 2077.5 (1.53×) · 2491.7 (1.83×) |
| large (H=1280, 20 heads, 36 layers) | 1024 | 523.4 | 681.2 (1.30×) · 729.2 (1.39×) | 679.4 (1.30×) · 727.9 (1.39×) | 677.2 (1.30×) · 727.1 (1.39×) |
| large (H=1280, 20 heads, 36 layers) | 4096 | 2756.1 | 4107.9 (1.49×) · 4864.2 (1.77×) | 4110.8 (1.49×) · 4869.4 (1.77×) | 4110.6 (1.49×) · 4863.8 (1.77×) |
| xl (H=1600, 25 heads, 48 layers) | 1024 | 1052.9 | 1330.6 (1.25×) · 1374.2 (1.30×) | 1324.6 (1.26×) · 1368.0 (1.30×) | 1319.4 (1.26×) · 1363.6 (1.30×) |
| xl (H=1600, 25 heads, 48 layers) | 4096 | 5263.9 | 7525.3 (1.43×) · 8919.9 (1.70×) | 7524.6 (1.43×) · 8919.0 (1.69×) | 7527.6 (1.43×) · 8909.9 (1.69×) |

## Chunks per MLP operand, 1 MiB / 2 MiB / 4 MiB class

| model | tokens | hidden state S×H | fc weight 4H×H | fc activation S×4H |
|---|---:|---|---|---|
| small | 1024 | 3 / 2 / 1 | 9 / 5 / 3 | 12 / 6 / 3 |
| small | 4096 | 12 / 6 / 3 | 9 / 5 / 3 | 48 / 24 / 12 |
| medium | 1024 | 4 / 2 / 1 | 16 / 8 / 4 | 16 / 8 / 4 |
| medium | 4096 | 16 / 8 / 4 | 16 / 8 / 4 | 64 / 32 / 16 |
| large | 1024 | 5 / 3 / 2 | 25 / 13 / 7 | 20 / 10 / 5 |
| large | 4096 | 20 / 10 / 5 | 25 / 13 / 7 | 80 / 40 / 20 |
| xl | 1024 | 7 / 4 / 2 | 40 / 20 / 10 | 25 / 13 / 7 |
| xl | 4096 | 25 / 13 / 7 | 40 / 20 / 10 | 100 / 50 / 25 |

## Per row, 1024 tokens/step


**GPT-2 small (H=768, 12 heads, 12 layers), 1024 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 1 MiB | chunked 1 MiB | staged 2 MiB | chunked 2 MiB | staged 4 MiB | chunked 4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 1024, 768, 768 × 1 | 48 | 158 | 194 (1.23×) | 172 (1.09×) | 192 (1.22×) | 172 (1.09×) | 156 (0.99×) | 171 (1.09×) |
| `qkvO_fwd` | 1024, 768, 768 × 1 | 48 | 136 | 173 (1.26×) | 173 (1.27×) | 171 (1.25×) | 172 (1.26×) | 135 (0.99×) | 174 (1.28×) |
| `qkvO_wgrad` | 768, 768, 1024 × 1 | 48 | 136 | 168 (1.24×) | 218 (1.61×) | 168 (1.24×) | 218 (1.60×) | 137 (1.01×) | 217 (1.60×) |
| `attn_ctx` | 1024, 64, 1024 × 12 | 24 | 233 | 454 (1.95×) | 396 (1.70×) | 453 (1.94×) | 396 (1.70×) | 417 (1.79×) | 385 (1.65×) |
| `attn_dKV` | 1024, 64, 1024 × 12 | 24 | 216 | 438 (2.03×) | 315 (1.46×) | 438 (2.03×) | 315 (1.46×) | 401 (1.85×) | 308 (1.43×) |
| `attn_scores` | 1024, 1024, 64 × 12 | 24 | 225 | 440 (1.96×) | 900 (4.00×) | 440 (1.95×) | 903 (4.02×) | 413 (1.84×) | 889 (3.95×) |
| `mlp_fc_dgrad` | 1024, 3072, 768 × 1 | 12 | 462 | 559 (1.21×) | 479 (1.04×) | 559 (1.21×) | 479 (1.04×) | 547 (1.18×) | 477 (1.03×) |
| `mlp_fc_fwd` | 1024, 3072, 768 × 1 | 12 | 471 | 571 (1.21×) | 514 (1.09×) | 571 (1.21×) | 514 (1.09×) | 556 (1.18×) | 482 (1.02×) |
| `mlp_fc_wgrad` | 768, 3072, 1024 × 1 | 12 | 506 | 601 (1.19×) | 651 (1.29×) | 601 (1.19×) | 651 (1.29×) | 588 (1.16×) | 475 (0.94×) |
| `mlp_proj_dgrad` | 1024, 768, 3072 × 1 | 12 | 464 | 559 (1.21×) | 625 (1.35×) | 559 (1.20×) | 625 (1.35×) | 545 (1.18×) | 482 (1.04×) |
| `mlp_proj_fwd` | 1024, 768, 3072 × 1 | 12 | 495 | 587 (1.19×) | 630 (1.27×) | 587 (1.19×) | 631 (1.28×) | 575 (1.16×) | 514 (1.04×) |
| `mlp_proj_wgrad` | 3072, 768, 1024 × 1 | 12 | 460 | 559 (1.22×) | 651 (1.42×) | 559 (1.22×) | 651 (1.42×) | 547 (1.19×) | 475 (1.03×) |
| `lm_head_dgrad` | 1024, 768, 50257 × 1 | 1 | 7182 | 8526 (1.19×) | n/a | 8526 (1.19×) | n/a | 8517 (1.19×) | n/a |
| `lm_head_fwd` | 1024, 50257, 768 × 1 | 1 | 7239 | 8622 (1.19×) | n/a | 8578 (1.18×) | n/a | 8576 (1.18×) | n/a |
| `lm_head_wgrad` | 50257, 768, 1024 × 1 | 1 | 6864 | 8261 (1.20×) | n/a | 8260 (1.20×) | n/a | 8187 (1.19×) | n/a |
| **per training step (ms)** | | | **92.4** | **124.3 (1.34×)** | **133.7 (1.45×)** | **124.0 (1.34×)** | **133.7 (1.45×)** | **115.7 (1.25×)** | **125.1 (1.35×)** |

No chunked route at 1 MiB: `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.
No chunked route at 2 MiB: `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.
No chunked route at 4 MiB: `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.

**GPT-2 medium (H=1024, 16 heads, 24 layers), 1024 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 1 MiB | chunked 1 MiB | staged 2 MiB | chunked 2 MiB | staged 4 MiB | chunked 4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 1024, 1024, 1024 × 1 | 96 | 218 | 266 (1.22×) | 223 (1.02×) | 266 (1.22×) | 223 (1.02×) | 217 (1.00×) | 220 (1.01×) |
| `qkvO_fwd` | 1024, 1024, 1024 × 1 | 96 | 235 | 284 (1.21×) | 228 (0.97×) | 283 (1.21×) | 228 (0.97×) | 233 (0.99×) | 225 (0.96×) |
| `qkvO_wgrad` | 1024, 1024, 1024 × 1 | 96 | 215 | 264 (1.23×) | 220 (1.02×) | 263 (1.22×) | 220 (1.02×) | 214 (1.00×) | 217 (1.01×) |
| `attn_ctx` | 1024, 64, 1024 × 16 | 48 | 256 | 557 (2.18×) | 512 (2.00×) | 555 (2.17×) | 512 (2.00×) | 508 (1.98×) | 498 (1.95×) |
| `attn_dKV` | 1024, 64, 1024 × 16 | 48 | 221 | 520 (2.36×) | 408 (1.85×) | 518 (2.35×) | 408 (1.85×) | 469 (2.13×) | 399 (1.81×) |
| `attn_scores` | 1024, 1024, 64 × 16 | 48 | 295 | 580 (1.96×) | 1183 (4.01×) | 580 (1.96×) | 1180 (4.00×) | 546 (1.85×) | 1177 (3.99×) |
| `mlp_fc_dgrad` | 1024, 4096, 1024 × 1 | 24 | 798 | 944 (1.18×) | 835 (1.05×) | 944 (1.18×) | 835 (1.05×) | 927 (1.16×) | 826 (1.03×) |
| `mlp_fc_fwd` | 1024, 4096, 1024 × 1 | 24 | 822 | 971 (1.18×) | 899 (1.09×) | 969 (1.18×) | 899 (1.09×) | 948 (1.15×) | 840 (1.02×) |
| `mlp_fc_wgrad` | 1024, 4096, 1024 × 1 | 24 | 788 | 935 (1.19×) | 863 (1.10×) | 934 (1.19×) | 864 (1.10×) | 916 (1.16×) | 821 (1.04×) |
| `mlp_proj_dgrad` | 1024, 1024, 4096 × 1 | 24 | 814 | 958 (1.18×) | 832 (1.02×) | 956 (1.17×) | 832 (1.02×) | 934 (1.15×) | 793 (0.97×) |
| `mlp_proj_fwd` | 1024, 1024, 4096 × 1 | 24 | 879 | 1022 (1.16×) | 862 (0.98×) | 1020 (1.16×) | 862 (0.98×) | 999 (1.14×) | 825 (0.94×) |
| `mlp_proj_wgrad` | 4096, 1024, 1024 × 1 | 24 | 788 | 936 (1.19×) | 865 (1.10×) | 935 (1.19×) | 868 (1.10×) | 918 (1.16×) | 824 (1.05×) |
| `lm_head_dgrad` | 1024, 1024, 50257 × 1 | 1 | 9581 | 11103 (1.16×) | n/a | 11103 (1.16×) | n/a | 11088 (1.16×) | n/a |
| `lm_head_fwd` | 1024, 50257, 1024 × 1 | 1 | 9520 | 11076 (1.16×) | n/a | 11076 (1.16×) | n/a | 11047 (1.16×) | n/a |
| `lm_head_wgrad` | 50257, 1024, 1024 × 1 | 1 | 9115 | 10705 (1.17×) | n/a | 10703 (1.17×) | n/a | 10626 (1.17×) | n/a |
| **per training step (ms)** | | | **246.7** | **328.9 (1.33×)** | **322.0 (1.31×)** | **328.4 (1.33×)** | **322.0 (1.31×)** | **305.1 (1.24×)** | **314.2 (1.27×)** |

No chunked route at 1 MiB: `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.
No chunked route at 2 MiB: `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.
No chunked route at 4 MiB: `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.

**GPT-2 large (H=1280, 20 heads, 36 layers), 1024 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 1 MiB | chunked 1 MiB | staged 2 MiB | chunked 2 MiB | staged 4 MiB | chunked 4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 1024, 1280, 1280 × 1 | 144 | 328 | 393 (1.20×) | 526 (1.60×) | 395 (1.20×) | 524 (1.60×) | 394 (1.20×) | 520 (1.59×) |
| `qkvO_fwd` | 1024, 1280, 1280 × 1 | 144 | 353 | 419 (1.19×) | 546 (1.55×) | 418 (1.19×) | 545 (1.55×) | 418 (1.18×) | 543 (1.54×) |
| `qkvO_wgrad` | 1280, 1280, 1024 × 1 | 144 | 315 | 387 (1.23×) | 434 (1.38×) | 385 (1.22×) | 434 (1.38×) | 384 (1.22×) | 434 (1.38×) |
| `attn_ctx` | 1024, 64, 1024 × 20 | 72 | 358 | 722 (2.02×) | 633 (1.77×) | 720 (2.01×) | 633 (1.77×) | 717 (2.00×) | 633 (1.77×) |
| `attn_dKV` | 1024, 64, 1024 × 20 | 72 | 316 | 689 (2.18×) | 504 (1.60×) | 686 (2.17×) | 504 (1.60×) | 683 (2.17×) | 504 (1.60×) |
| `attn_scores` | 1024, 1024, 64 × 20 | 72 | 366 | 721 (1.97×) | 1467 (4.00×) | 719 (1.96×) | 1469 (4.01×) | 719 (1.96×) | 1472 (4.02×) |
| `mlp_fc_dgrad` | 1024, 5120, 1280 × 1 | 36 | 1240 | 1443 (1.16×) | 1276 (1.03×) | 1442 (1.16×) | 1276 (1.03×) | 1443 (1.16×) | 1279 (1.03×) |
| `mlp_fc_fwd` | 1024, 5120, 1280 × 1 | 36 | 1277 | 1486 (1.16×) | 1352 (1.06×) | 1482 (1.16×) | 1348 (1.06×) | 1473 (1.15×) | 1350 (1.06×) |
| `mlp_fc_wgrad` | 1280, 5120, 1024 × 1 | 36 | 1198 | 1413 (1.18×) | 1294 (1.08×) | 1401 (1.17×) | 1292 (1.08×) | 1388 (1.16×) | 1295 (1.08×) |
| `mlp_proj_dgrad` | 1024, 1280, 5120 × 1 | 36 | 1249 | 1444 (1.16×) | 1313 (1.05×) | 1444 (1.16×) | 1300 (1.04×) | 1445 (1.16×) | 1286 (1.03×) |
| `mlp_proj_fwd` | 1024, 1280, 5120 × 1 | 36 | 1345 | 1544 (1.15×) | 1377 (1.02×) | 1541 (1.15×) | 1376 (1.02×) | 1540 (1.15×) | 1375 (1.02×) |
| `mlp_proj_wgrad` | 5120, 1280, 1024 × 1 | 36 | 1198 | 1413 (1.18×) | 1292 (1.08×) | 1401 (1.17×) | 1289 (1.08×) | 1388 (1.16×) | 1290 (1.08×) |
| `lm_head_dgrad` | 1024, 1280, 50257 × 1 | 1 | 11497 | 13230 (1.15×) | n/a | 13229 (1.15×) | n/a | 13228 (1.15×) | n/a |
| `lm_head_fwd` | 1024, 50257, 1280 × 1 | 1 | 11961 | 13732 (1.15×) | n/a | 13777 (1.15×) | n/a | 13767 (1.15×) | n/a |
| `lm_head_wgrad` | 50257, 1280, 1024 × 1 | 1 | 11406 | 13243 (1.16×) | n/a | 13167 (1.15×) | n/a | 13146 (1.15×) | n/a |
| **per training step (ms)** | | | **523.4** | **681.2 (1.30×)** | **729.2 (1.39×)** | **679.4 (1.30×)** | **727.9 (1.39×)** | **677.2 (1.29×)** | **727.1 (1.39×)** |

No chunked route at 1 MiB: `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.
No chunked route at 2 MiB: `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.
No chunked route at 4 MiB: `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.

**GPT-2 xl (H=1600, 25 heads, 48 layers), 1024 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 1 MiB | chunked 1 MiB | staged 2 MiB | chunked 2 MiB | staged 4 MiB | chunked 4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 1024, 1600, 1600 × 1 | 192 | 519 | 613 (1.18×) | n/a | 607 (1.17×) | n/a | 606 (1.17×) | n/a |
| `qkvO_fwd` | 1024, 1600, 1600 × 1 | 192 | 634 | 727 (1.15×) | n/a | 722 (1.14×) | n/a | 720 (1.14×) | n/a |
| `qkvO_wgrad` | 1600, 1600, 1024 × 1 | 192 | 502 | 594 (1.18×) | n/a | 594 (1.18×) | n/a | 591 (1.18×) | n/a |
| `attn_ctx` | 1024, 64, 1024 × 25 | 96 | 470 | 961 (2.05×) | 758 (1.61×) | 962 (2.05×) | 756 (1.61×) | 960 (2.04×) | 755 (1.61×) |
| `attn_dKV` | 1024, 64, 1024 × 25 | 96 | 415 | 880 (2.12×) | 606 (1.46×) | 876 (2.11×) | 602 (1.45×) | 873 (2.10×) | 600 (1.45×) |
| `attn_scores` | 1024, 1024, 64 × 25 | 96 | 455 | 897 (1.97×) | 1829 (4.02×) | 895 (1.97×) | 1828 (4.01×) | 894 (1.96×) | 1831 (4.02×) |
| `mlp_fc_dgrad` | 1024, 6400, 1600 × 1 | 48 | 1880 | 2185 (1.16×) | n/a | 2163 (1.15×) | n/a | 2142 (1.14×) | n/a |
| `mlp_fc_fwd` | 1024, 6400, 1600 × 1 | 48 | 1906 | 2204 (1.16×) | n/a | 2189 (1.15×) | n/a | 2176 (1.14×) | n/a |
| `mlp_fc_wgrad` | 1600, 6400, 1024 × 1 | 48 | 1875 | 2152 (1.15×) | n/a | 2152 (1.15×) | n/a | 2152 (1.15×) | n/a |
| `mlp_proj_dgrad` | 1024, 1600, 6400 × 1 | 48 | 2050 | 2357 (1.15×) | n/a | 2327 (1.14×) | n/a | 2291 (1.12×) | n/a |
| `mlp_proj_fwd` | 1024, 1600, 6400 × 1 | 48 | 2038 | 2316 (1.14×) | n/a | 2316 (1.14×) | n/a | 2313 (1.13×) | n/a |
| `mlp_proj_wgrad` | 6400, 1600, 1024 × 1 | 48 | 1959 | 2239 (1.14×) | n/a | 2238 (1.14×) | n/a | 2238 (1.14×) | n/a |
| `lm_head_dgrad` | 1024, 1600, 50257 × 1 | 1 | 15284 | 17275 (1.13×) | n/a | 17275 (1.13×) | n/a | 17274 (1.13×) | n/a |
| `lm_head_fwd` | 1024, 50257, 1600 × 1 | 1 | 14816 | 16844 (1.14×) | n/a | 16855 (1.14×) | n/a | 16894 (1.14×) | n/a |
| `lm_head_wgrad` | 50257, 1600, 1024 × 1 | 1 | 14383 | 16423 (1.14×) | n/a | 16420 (1.14×) | n/a | 16357 (1.14×) | n/a |
| **per training step (ms)** | | | **1052.9** | **1330.6 (1.26×)** | **1374.2 (1.31×)** | **1324.6 (1.26×)** | **1368.0 (1.30×)** | **1319.4 (1.25×)** | **1363.6 (1.30×)** |

No chunked route at 1 MiB: `qkvO_dgrad`, `qkvO_fwd`, `qkvO_wgrad`, `mlp_fc_dgrad`, `mlp_fc_fwd`, `mlp_fc_wgrad`, `mlp_proj_dgrad`, `mlp_proj_fwd`, `mlp_proj_wgrad`, `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.
No chunked route at 2 MiB: `qkvO_dgrad`, `qkvO_fwd`, `qkvO_wgrad`, `mlp_fc_dgrad`, `mlp_fc_fwd`, `mlp_fc_wgrad`, `mlp_proj_dgrad`, `mlp_proj_fwd`, `mlp_proj_wgrad`, `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.
No chunked route at 4 MiB: `qkvO_dgrad`, `qkvO_fwd`, `qkvO_wgrad`, `mlp_fc_dgrad`, `mlp_fc_fwd`, `mlp_fc_wgrad`, `mlp_proj_dgrad`, `mlp_proj_fwd`, `mlp_proj_wgrad`, `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.

## Per row, 4096 tokens/step


**GPT-2 small (H=768, 12 heads, 12 layers), 4096 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 1 MiB | chunked 1 MiB | staged 2 MiB | chunked 2 MiB | staged 4 MiB | chunked 4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 4096, 768, 768 × 1 | 48 | 461 | 567 (1.23×) | 483 (1.05×) | 567 (1.23×) | 483 (1.05×) | 560 (1.22×) | n/a |
| `qkvO_fwd` | 4096, 768, 768 × 1 | 48 | 471 | 582 (1.23×) | 516 (1.10×) | 580 (1.23×) | 516 (1.09×) | 565 (1.20×) | n/a |
| `qkvO_wgrad` | 768, 768, 4096 × 1 | 48 | 486 | 588 (1.21×) | 821 (1.69×) | 588 (1.21×) | 817 (1.68×) | 580 (1.19×) | 806 (1.66×) |
| `attn_ctx` | 4096, 64, 4096 × 12 | 24 | 2950 | 6245 (2.12×) | 5734 (1.94×) | 6236 (2.11×) | 5733 (1.94×) | 6258 (2.12×) | 5734 (1.94×) |
| `attn_dKV` | 4096, 64, 4096 × 12 | 24 | 2433 | 5725 (2.35×) | 4583 (1.88×) | 5714 (2.35×) | 4584 (1.88×) | 5713 (2.35×) | 4585 (1.88×) |
| `attn_scores` | 4096, 4096, 64 × 12 | 24 | 3413 | 6461 (1.89×) | 15434 (4.52×) | 6459 (1.89×) | 15472 (4.53×) | 6458 (1.89×) | 15421 (4.52×) |
| `mlp_fc_dgrad` | 4096, 3072, 768 × 1 | 12 | 1730 | 2000 (1.16×) | 1868 (1.08×) | 2000 (1.16×) | 1866 (1.08×) | 1999 (1.16×) | 1866 (1.08×) |
| `mlp_fc_fwd` | 4096, 3072, 768 × 1 | 12 | 1835 | 2106 (1.15×) | 2048 (1.12×) | 2105 (1.15×) | 2048 (1.12×) | 2106 (1.15×) | 2045 (1.11×) |
| `mlp_fc_wgrad` | 768, 3072, 4096 × 1 | 12 | 1716 | 1995 (1.16×) | 1761 (1.03×) | 1995 (1.16×) | 1755 (1.02×) | 1995 (1.16×) | 1756 (1.02×) |
| `mlp_proj_dgrad` | 4096, 768, 3072 × 1 | 12 | 1792 | 2065 (1.15×) | 1854 (1.03×) | 2064 (1.15×) | 1848 (1.03×) | 2065 (1.15×) | 1849 (1.03×) |
| `mlp_proj_fwd` | 4096, 768, 3072 × 1 | 12 | 1823 | 2108 (1.16×) | 1880 (1.03×) | 2104 (1.15×) | 1877 (1.03×) | 2098 (1.15×) | 1877 (1.03×) |
| `mlp_proj_wgrad` | 3072, 768, 4096 × 1 | 12 | 1773 | 2050 (1.16×) | 1755 (0.99×) | 2050 (1.16×) | 1757 (0.99×) | 2050 (1.16×) | 1756 (0.99×) |
| `lm_head_dgrad` | 4096, 768, 50257 × 1 | 1 | 28594 | 32428 (1.13×) | n/a | 32428 (1.13×) | n/a | 32428 (1.13×) | n/a |
| `lm_head_fwd` | 4096, 50257, 768 × 1 | 1 | 28508 | 32149 (1.13×) | n/a | 32203 (1.13×) | n/a | 32185 (1.13×) | n/a |
| `lm_head_wgrad` | 50257, 768, 4096 × 1 | 1 | 27049 | 30928 (1.14×) | n/a | 30927 (1.14×) | n/a | 30927 (1.14×) | n/a |
| **per training step (ms)** | | | **491.4** | **769.1 (1.57×)** | **934.9 (1.90×)** | **768.5 (1.56×)** | **935.5 (1.90×)** | **767.5 (1.56×)** | **939.8 (1.91×)** |

No chunked route at 1 MiB: `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.
No chunked route at 2 MiB: `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.
No chunked route at 4 MiB: `qkvO_dgrad`, `qkvO_fwd`, `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.

**GPT-2 medium (H=1024, 16 heads, 24 layers), 4096 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 1 MiB | chunked 1 MiB | staged 2 MiB | chunked 2 MiB | staged 4 MiB | chunked 4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 4096, 1024, 1024 × 1 | 96 | 798 | 945 (1.18×) | 831 (1.04×) | 945 (1.18×) | 834 (1.04×) | 928 (1.16×) | n/a |
| `qkvO_fwd` | 4096, 1024, 1024 × 1 | 96 | 823 | 969 (1.18×) | 894 (1.09×) | 971 (1.18×) | 894 (1.09×) | 952 (1.16×) | n/a |
| `qkvO_wgrad` | 1024, 1024, 4096 × 1 | 96 | 808 | 946 (1.17×) | 811 (1.00×) | 948 (1.17×) | 815 (1.01×) | 934 (1.16×) | 812 (1.01×) |
| `attn_ctx` | 4096, 64, 4096 × 16 | 48 | 3967 | 8311 (2.10×) | 7588 (1.91×) | 8331 (2.10×) | 7588 (1.91×) | 8313 (2.10×) | 7587 (1.91×) |
| `attn_dKV` | 4096, 64, 4096 × 16 | 48 | 3201 | 7575 (2.37×) | 6066 (1.90×) | 7575 (2.37×) | 6067 (1.90×) | 7576 (2.37×) | 6068 (1.90×) |
| `attn_scores` | 4096, 4096, 64 × 16 | 48 | 4545 | 8601 (1.89×) | 20616 (4.54×) | 8605 (1.89×) | 20646 (4.54×) | 8604 (1.89×) | 20574 (4.53×) |
| `mlp_fc_dgrad` | 4096, 4096, 1024 × 1 | 24 | 3097 | 3468 (1.12×) | 3264 (1.05×) | 3467 (1.12×) | 3264 (1.05×) | 3467 (1.12×) | 3264 (1.05×) |
| `mlp_fc_fwd` | 4096, 4096, 1024 × 1 | 24 | 3230 | 3602 (1.12×) | 3562 (1.10×) | 3600 (1.11×) | 3556 (1.10×) | 3602 (1.12×) | 3555 (1.10×) |
| `mlp_fc_wgrad` | 1024, 4096, 4096 × 1 | 24 | 3098 | 3487 (1.13×) | 3098 (1.00×) | 3487 (1.13×) | 3103 (1.00×) | 3486 (1.13×) | 3108 (1.00×) |
| `mlp_proj_dgrad` | 4096, 1024, 4096 × 1 | 24 | 3129 | 3518 (1.12×) | 3151 (1.01×) | 3518 (1.12×) | 3159 (1.01×) | 3518 (1.12×) | 3164 (1.01×) |
| `mlp_proj_fwd` | 4096, 1024, 4096 × 1 | 24 | 3210 | 3595 (1.12×) | 3205 (1.00×) | 3596 (1.12×) | 3207 (1.00×) | 3594 (1.12×) | 3209 (1.00×) |
| `mlp_proj_wgrad` | 4096, 1024, 4096 × 1 | 24 | 3098 | 3488 (1.13×) | 3097 (1.00×) | 3488 (1.13×) | 3104 (1.00×) | 3488 (1.13×) | 3127 (1.01×) |
| `lm_head_dgrad` | 4096, 1024, 50257 × 1 | 1 | 38119 | 42141 (1.11×) | n/a | 42142 (1.11×) | n/a | 42142 (1.11×) | n/a |
| `lm_head_fwd` | 4096, 50257, 1024 × 1 | 1 | 37623 | 41458 (1.10×) | n/a | 41449 (1.10×) | n/a | 41465 (1.10×) | n/a |
| `lm_head_wgrad` | 50257, 1024, 4096 × 1 | 1 | 36319 | 40395 (1.11×) | n/a | 40395 (1.11×) | n/a | 40395 (1.11×) | n/a |
| **per training step (ms)** | | | **1360.1** | **2081.7 (1.53×)** | **2477.5 (1.82×)** | **2083.2 (1.53×)** | **2479.9 (1.82×)** | **2077.5 (1.53×)** | **2491.7 (1.83×)** |

No chunked route at 1 MiB: `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.
No chunked route at 2 MiB: `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.
No chunked route at 4 MiB: `qkvO_dgrad`, `qkvO_fwd`, `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.

**GPT-2 large (H=1280, 20 heads, 36 layers), 4096 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 1 MiB | chunked 1 MiB | staged 2 MiB | chunked 2 MiB | staged 4 MiB | chunked 4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 4096, 1280, 1280 × 1 | 144 | 1202 | 1390 (1.16×) | 1289 (1.07×) | 1389 (1.16×) | 1289 (1.07×) | 1390 (1.16×) | 1289 (1.07×) |
| `qkvO_fwd` | 4096, 1280, 1280 × 1 | 144 | 1279 | 1467 (1.15×) | 1366 (1.07×) | 1467 (1.15×) | 1367 (1.07×) | 1468 (1.15×) | 1366 (1.07×) |
| `qkvO_wgrad` | 1280, 1280, 4096 × 1 | 144 | 1205 | 1385 (1.15×) | 1614 (1.34×) | 1388 (1.15×) | 1613 (1.34×) | 1389 (1.15×) | 1613 (1.34×) |
| `attn_ctx` | 4096, 64, 4096 × 20 | 72 | 4970 | 10439 (2.10×) | 9123 (1.84×) | 10453 (2.10×) | 9131 (1.84×) | 10426 (2.10×) | 9128 (1.84×) |
| `attn_dKV` | 4096, 64, 4096 × 20 | 72 | 3983 | 9456 (2.37×) | 7277 (1.83×) | 9457 (2.37×) | 7277 (1.83×) | 9457 (2.37×) | 7277 (1.83×) |
| `attn_scores` | 4096, 4096, 64 × 20 | 72 | 5679 | 10748 (1.89×) | 25772 (4.54×) | 10750 (1.89×) | 25810 (4.55×) | 10750 (1.89×) | 25730 (4.53×) |
| `mlp_fc_dgrad` | 4096, 5120, 1280 × 1 | 36 | 4689 | 5177 (1.10×) | 4842 (1.03×) | 5176 (1.10×) | 4846 (1.03×) | 5176 (1.10×) | 4846 (1.03×) |
| `mlp_fc_fwd` | 4096, 5120, 1280 × 1 | 36 | 4830 | 5325 (1.10×) | 5153 (1.07×) | 5324 (1.10×) | 5156 (1.07×) | 5325 (1.10×) | 5155 (1.07×) |
| `mlp_fc_wgrad` | 1280, 5120, 4096 × 1 | 36 | 4714 | 5208 (1.10×) | 4810 (1.02×) | 5216 (1.11×) | 4819 (1.02×) | 5218 (1.11×) | 4827 (1.02×) |
| `mlp_proj_dgrad` | 4096, 1280, 5120 × 1 | 36 | 4684 | 5185 (1.11×) | 4918 (1.05×) | 5185 (1.11×) | 4933 (1.05×) | 5185 (1.11×) | 4938 (1.05×) |
| `mlp_proj_fwd` | 4096, 1280, 5120 × 1 | 36 | 5037 | 5523 (1.10×) | 4942 (0.98×) | 5540 (1.10×) | 4944 (0.98×) | 5573 (1.11×) | 4945 (0.98×) |
| `mlp_proj_wgrad` | 5120, 1280, 4096 × 1 | 36 | 4718 | 5211 (1.10×) | 4805 (1.02×) | 5222 (1.11×) | 4825 (1.02×) | 5222 (1.11×) | 4827 (1.02×) |
| `lm_head_dgrad` | 4096, 1280, 50257 × 1 | 1 | 47643 | 51853 (1.09×) | n/a | 51853 (1.09×) | n/a | 51853 (1.09×) | n/a |
| `lm_head_fwd` | 4096, 50257, 1280 × 1 | 1 | 47225 | 51204 (1.08×) | n/a | 51245 (1.09×) | n/a | 51258 (1.09×) | n/a |
| `lm_head_wgrad` | 50257, 1280, 4096 × 1 | 1 | 44818 | 49102 (1.10×) | n/a | 49102 (1.10×) | n/a | 49103 (1.10×) | n/a |
| **per training step (ms)** | | | **2756.1** | **4107.9 (1.49×)** | **4864.2 (1.76×)** | **4110.8 (1.49×)** | **4869.4 (1.77×)** | **4110.6 (1.49×)** | **4863.8 (1.76×)** |

No chunked route at 1 MiB: `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.
No chunked route at 2 MiB: `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.
No chunked route at 4 MiB: `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.

**GPT-2 xl (H=1600, 25 heads, 48 layers), 4096 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 1 MiB | chunked 1 MiB | staged 2 MiB | chunked 2 MiB | staged 4 MiB | chunked 4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 4096, 1600, 1600 × 1 | 192 | 1958 | 2196 (1.12×) | n/a | 2196 (1.12×) | n/a | 2195 (1.12×) | n/a |
| `qkvO_fwd` | 4096, 1600, 1600 × 1 | 192 | 2094 | 2332 (1.11×) | n/a | 2332 (1.11×) | n/a | 2327 (1.11×) | n/a |
| `qkvO_wgrad` | 1600, 1600, 4096 × 1 | 192 | 1882 | 2122 (1.13×) | n/a | 2119 (1.13×) | n/a | 2122 (1.13×) | n/a |
| `attn_ctx` | 4096, 64, 4096 × 25 | 96 | 6387 | 13234 (2.07×) | 11447 (1.79×) | 13238 (2.07×) | 11448 (1.79×) | 13272 (2.08×) | 11448 (1.79×) |
| `attn_dKV` | 4096, 64, 4096 × 25 | 96 | 4776 | 11618 (2.43×) | 9144 (1.91×) | 11620 (2.43×) | 9144 (1.91×) | 11618 (2.43×) | 9142 (1.91×) |
| `attn_scores` | 4096, 4096, 64 × 25 | 96 | 7096 | 13431 (1.89×) | 32219 (4.54×) | 13433 (1.89×) | 32223 (4.54×) | 13434 (1.89×) | 32133 (4.53×) |
| `mlp_fc_dgrad` | 4096, 6400, 1600 × 1 | 48 | 7395 | 8030 (1.09×) | n/a | 8029 (1.09×) | n/a | 8028 (1.09×) | n/a |
| `mlp_fc_fwd` | 4096, 6400, 1600 × 1 | 48 | 7559 | 8219 (1.09×) | n/a | 8203 (1.09×) | n/a | 8207 (1.09×) | n/a |
| `mlp_fc_wgrad` | 1600, 6400, 4096 × 1 | 48 | 7234 | 7890 (1.09×) | n/a | 7890 (1.09×) | n/a | 7890 (1.09×) | n/a |
| `mlp_proj_dgrad` | 4096, 1600, 6400 × 1 | 48 | 7688 | 8338 (1.08×) | n/a | 8339 (1.08×) | n/a | 8338 (1.08×) | n/a |
| `mlp_proj_fwd` | 4096, 1600, 6400 × 1 | 48 | 8097 | 8761 (1.08×) | n/a | 8764 (1.08×) | n/a | 8765 (1.08×) | n/a |
| `mlp_proj_wgrad` | 6400, 1600, 4096 × 1 | 48 | 7734 | 8392 (1.09×) | n/a | 8392 (1.09×) | n/a | 8392 (1.09×) | n/a |
| `lm_head_dgrad` | 4096, 1600, 50257 × 1 | 1 | 61962 | 66399 (1.07×) | n/a | 66401 (1.07×) | n/a | 66402 (1.07×) | n/a |
| `lm_head_fwd` | 4096, 50257, 1600 × 1 | 1 | 58656 | 62966 (1.07×) | n/a | 62917 (1.07×) | n/a | 62932 (1.07×) | n/a |
| `lm_head_wgrad` | 50257, 1600, 4096 × 1 | 1 | 57212 | 61739 (1.08×) | n/a | 61738 (1.08×) | n/a | 61736 (1.08×) | n/a |
| **per training step (ms)** | | | **5263.9** | **7525.3 (1.43×)** | **8919.9 (1.69×)** | **7524.6 (1.43×)** | **8919.0 (1.69×)** | **7527.6 (1.43×)** | **8909.9 (1.69×)** |

No chunked route at 1 MiB: `qkvO_dgrad`, `qkvO_fwd`, `qkvO_wgrad`, `mlp_fc_dgrad`, `mlp_fc_fwd`, `mlp_fc_wgrad`, `mlp_proj_dgrad`, `mlp_proj_fwd`, `mlp_proj_wgrad`, `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.
No chunked route at 2 MiB: `qkvO_dgrad`, `qkvO_fwd`, `qkvO_wgrad`, `mlp_fc_dgrad`, `mlp_fc_fwd`, `mlp_fc_wgrad`, `mlp_proj_dgrad`, `mlp_proj_fwd`, `mlp_proj_wgrad`, `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.
No chunked route at 4 MiB: `qkvO_dgrad`, `qkvO_fwd`, `qkvO_wgrad`, `mlp_fc_dgrad`, `mlp_fc_fwd`, `mlp_fc_wgrad`, `mlp_proj_dgrad`, `mlp_proj_fwd`, `mlp_proj_wgrad`, `lm_head_dgrad`, `lm_head_fwd`, `lm_head_wgrad`.

## Reading the numbers

- **Staging costs 1.24–1.34× cuBLAS per step at 1024 tokens and
  1.43–1.57× at 4096 tokens**, whatever the class. On the attention rows
  it is about 2× (the per-head score matrices are whole chunks that must
  be gathered and scattered), on the MLP, projection and lm-head rows
  1.07–1.24×, except the projection rows whose every operand is a single
  chunk on the 4 MiB class, where staging is plain cuBLAS.
- **The class matters to staging only when an operand stops being
  chunked.** The 1 MiB and 2 MiB classes give the same picture. At 4 MiB
  the hidden state and the fc weight of GPT-2 small and medium at 1024
  tokens fit one chunk, so fewer bytes are copied: staged drops from 1.34×
  to 1.25× of cuBLAS per step for small and from 1.33× to 1.24× for
  medium, and on their projection rows staging is plain cuBLAS.
- **On the MLP rows the chunked route beats staging nearly everywhere.**
  Of the 36 MLP rows of GPT-2 small, medium and large it is faster than
  staged on 32 at 1 MiB, 32 at 2 MiB and all 36 at 4 MiB; the exceptions
  are GPT-2 small's rows at 1024 tokens. Summed over the six MLP rows of a
  step, chunked stands at 1.01–1.05× cuBLAS for medium and large on every
  class and at 4096 tokens for small too, against 1.10–1.18× for staging.
- **Against cuBLAS the chunked route is close on the MLP rows, not on
  par.** Per row it is within 2 % of cuBLAS on 8, 8 and 9 of those 36 rows
  at 1, 2 and 4 MiB, within 10 % on 30, 29 and 34, and faster than cuBLAS
  on 6, 4 and 5. The worst rows are GPT-2 small at 1024 tokens on the
  1 MiB and 2 MiB classes (1.27–1.42×: every operand is chunked there and
  the row runs the core bridge) and `mlp_fc_fwd` at 4096 tokens for small
  and medium (1.10–1.12× on every class).
- **The projection rows favor the chunked route only for GPT-2 medium**,
  where it is at 0.97–1.09× cuBLAS against 1.17–1.22× for staging. For
  small and large it is behind: `qkvO_wgrad` at 1.6–1.7× for small and
  1.3–1.4× for large (staging 1.2×), and large's `qkvO_fwd` and `qkvO_dgrad`
  at 1024 tokens at 1.5–1.6×. On the 4 MiB class the forward and dgrad
  rows of small and medium at 4096 tokens have no chunked route (one
  operand is a single chunk): they stay at 1.16–1.22× staged.
- **Attention is lost by both routes, the chunked one further.** The
  three batched attention rows cost staging 1.8–2.1× cuBLAS per step and
  the chunked route 2.4–2.9×.
- **No chunked route on the lm head and on XL's MLP and projection rows.**
  The 50257-wide vocabulary and H=1600 are outside the native family's
  shape law, so those rows are staged only (1.07–1.20×).

**What this says.** Over a whole training step the chunked route is far
behind contiguous cuBLAS: 1.28–1.45× at 1024 tokens and 1.69–1.91× at
4096 tokens, and slower than staging at 4096 tokens. That total mixes
rows where the native family runs with rows where it has no route and
staging is counted instead. Where it does run, the picture depends on
the family (rows counted once per chunk class, 360 in all):

| family | rows with a chunked route | of those, faster than staged | chunked / cuBLAS, median (range) |
|---|---:|---:|---|
| MLP | 108 of 144 | 100 | 1.04× (0.94–1.42×) |
| projection | 50 of 72 | 26 | 1.09× (0.96–1.69×) |
| attention | 72 of 72 | 48 | 1.91× (1.43–4.54×) |
| lm head | 0 of 72 | – | no chunked route |

The MLP GEMMs of GPT-2 small, medium and large are the one family where
the chunked route is competitive, and only in aggregate: summed per
model they stand at 1.01–1.05× cuBLAS, except small at 1024 tokens
(1.24× on the 1 MiB and 2 MiB classes, 1.02× on 4 MiB). Per row, 9 to 18
of those 36 rows on a class are more than 5 % behind cuBLAS, the worst
1.42×. GPT-2 XL's MLP and projection rows and every lm-head row have no
chunked route. Production routing is unchanged: the wrapper still sends
these calls to plain cuBLAS or to staging.

## Reproduce

```
# one nnhaul engine per class: configure it with NNHAUL_CHUNK_SIZE_MB=1, 2
# and 4, install each, and build nntile against each (the staged body's
# copy kernels are compiled for the engine's class)
cmake --build build --target bench_gemm_chunked_gpt2
build/nntile/tests/kernel/bench_gemm_chunked_gpt2 1 > routes_1mib.csv
build/nntile/tests/kernel/bench_gemm_chunked_gpt2 2 > routes_2mib.csv
build/nntile/tests/kernel/bench_gemm_chunked_gpt2 4 > routes_4mib.csv
python3 torch_nntile/tools/render_chunk_class_tables.py \
    routes_1mib.csv routes_2mib.csv routes_4mib.csv > tables.md
```

The benchmark's first argument is the chunk class in MiB, which must be
the build's own for the staged routes to exist; with no arguments it uses
the engine's class, 1024 and 4096 tokens and all four models. The
committed files are `gemm_routes_chunk_class_1mib.csv`,
`gemm_routes_chunk_class_2mib.csv` and `gemm_routes_chunk_class_4mib.csv`.
