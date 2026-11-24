# GEMM routes by chunk class: 1, 2 and 4 MiB (GPT-2 S to XL)

What the three GEMM routes cost on every GEMM of a GPT-2 training step
when the engine's chunk class is 1, 2 or 4 MiB: contiguous cuBLAS, the
workspace-staged route with its copies, and the native chunked family of
`nntile/src/kernel/gemm_chunked/`. Measured with `bench_gemm_chunked_gpt2`
(`nntile/tests/kernel/gemm_chunked_gpt2_bench.cc`) on 2026-11-24 at commit
`c8f07d69` of branch `gemm-chunked`. Every table below is the output of
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
   production ownership policy says). No workspace. `n/a` would mark a
   row the family cannot run; with the `tails` piece there is none on
   these rows.

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
are sums of calls × median: GEMM time only, the chunked total over the
native route of every row.

**Conditions.** NVIDIA GeForce RTX 2080 Ti (sm_75), CUDA 12.4, cuBLAS
12.4.5.8, CUTLASS 2.11.0; one process on the GPU, SM clock pinned at
1350 MHz. Median of at least 50 timed iterations (CUDA event pairs, five
interleaved rounds). Every route was checked against the cuBLAS result on
the same data before timing: staged is bitwise identical on all 120 rows
of every class, the chunked route is within 1.2e-5 relative.

Three engine builds were measured, because the staged body's copy kernels
are compiled for one chunk class: for each class an nnhaul engine built
with that `NNHAUL_CHUNK_SIZE_MB` and an nntile build against it. The
native family takes the class at runtime. cuBLAS on flat operands does
not depend on the class; it was measured three times, once per build, and
repeated within 1 % on 104 of the 120 rows (largest spread
2.5 %): the tables show the mean, so a per-row difference of
a few percent between classes is inside the run-to-run variation.

## GEMM time per training step

| model | tokens/step | cuBLAS ms | 1 MiB class: staged · chunked | 2 MiB class: staged · chunked | 4 MiB class: staged · chunked |
|---|---:|---:|---|---|---|
| small (H=768, 12 heads, 12 layers) | 1024 | 92.3 | 124.2 (1.34×) · 140.9 (1.52×) | 123.8 (1.34×) · 140.4 (1.52×) | 115.7 (1.25×) · 131.8 (1.43×) |
| small (H=768, 12 heads, 12 layers) | 4096 | 491.5 | 768.3 (1.56×) · 952.4 (1.94×) | 768.1 (1.56×) · 952.3 (1.94×) | 767.6 (1.56×) · 955.0 (1.95×) |
| medium (H=1024, 16 heads, 24 layers) | 1024 | 246.8 | 329.9 (1.33×) · 328.6 (1.32×) | 327.4 (1.33×) · 326.8 (1.33×) | 305.1 (1.24×) · 319.5 (1.30×) |
| medium (H=1024, 16 heads, 24 layers) | 4096 | 1359.8 | 2082.6 (1.53×) · 2504.2 (1.84×) | 2082.5 (1.53×) · 2505.9 (1.84×) | 2078.1 (1.53×) · 2513.3 (1.85×) |
| large (H=1280, 20 heads, 36 layers) | 1024 | 522.5 | 679.6 (1.30×) · 746.1 (1.43×) | 677.6 (1.30×) · 744.6 (1.43×) | 677.3 (1.30×) · 744.3 (1.42×) |
| large (H=1280, 20 heads, 36 layers) | 4096 | 2757.4 | 4112.1 (1.49×) · 4901.9 (1.78×) | 4111.1 (1.49×) · 4904.0 (1.78×) | 4110.8 (1.49×) · 4896.7 (1.78×) |
| xl (H=1600, 25 heads, 48 layers) | 1024 | 1047.9 | 1321.7 (1.26×) · 1578.4 (1.50×) | 1319.4 (1.26×) · 1576.7 (1.50×) | 1319.4 (1.26×) · 1576.9 (1.51×) |
| xl (H=1600, 25 heads, 48 layers) | 4096 | 5264.7 | 7525.7 (1.43×) · 9712.1 (1.84×) | 7526.0 (1.43×) · 9709.9 (1.84×) | 7524.8 (1.43×) · 9698.0 (1.84×) |

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
| `qkvO_dgrad` | 1024, 768, 768 × 1 | 48 | 157 | 192 (1.23×) | 172 (1.10×) | 190 (1.21×) | 172 (1.10×) | 156 (1.00×) | 171 (1.09×) |
| `qkvO_fwd` | 1024, 768, 768 × 1 | 48 | 136 | 172 (1.27×) | 172 (1.27×) | 170 (1.25×) | 172 (1.27×) | 135 (0.99×) | 174 (1.28×) |
| `qkvO_wgrad` | 768, 768, 1024 × 1 | 48 | 136 | 168 (1.23×) | 218 (1.60×) | 169 (1.24×) | 217 (1.60×) | 137 (1.00×) | 217 (1.59×) |
| `attn_ctx` | 1024, 64, 1024 × 12 | 24 | 233 | 455 (1.95×) | 396 (1.70×) | 453 (1.94×) | 395 (1.70×) | 417 (1.79×) | 385 (1.65×) |
| `attn_dKV` | 1024, 64, 1024 × 12 | 24 | 216 | 440 (2.04×) | 315 (1.46×) | 436 (2.02×) | 315 (1.46×) | 401 (1.86×) | 308 (1.43×) |
| `attn_scores` | 1024, 1024, 64 × 12 | 24 | 225 | 440 (1.96×) | 900 (4.01×) | 439 (1.95×) | 901 (4.01×) | 414 (1.84×) | 889 (3.96×) |
| `mlp_fc_dgrad` | 1024, 3072, 768 × 1 | 12 | 462 | 559 (1.21×) | 479 (1.04×) | 558 (1.21×) | 478 (1.04×) | 547 (1.18×) | 477 (1.03×) |
| `mlp_fc_fwd` | 1024, 3072, 768 × 1 | 12 | 471 | 572 (1.21×) | 514 (1.09×) | 568 (1.21×) | 513 (1.09×) | 557 (1.18×) | 482 (1.02×) |
| `mlp_fc_wgrad` | 768, 3072, 1024 × 1 | 12 | 506 | 601 (1.19×) | 651 (1.29×) | 602 (1.19×) | 651 (1.29×) | 588 (1.16×) | 475 (0.94×) |
| `mlp_proj_dgrad` | 1024, 768, 3072 × 1 | 12 | 464 | 561 (1.21×) | 627 (1.35×) | 555 (1.20×) | 619 (1.34×) | 545 (1.18×) | 482 (1.04×) |
| `mlp_proj_fwd` | 1024, 768, 3072 × 1 | 12 | 494 | 589 (1.19×) | 633 (1.28×) | 584 (1.18×) | 625 (1.26×) | 576 (1.16×) | 515 (1.04×) |
| `mlp_proj_wgrad` | 3072, 768, 1024 × 1 | 12 | 460 | 559 (1.22×) | 651 (1.42×) | 559 (1.22×) | 651 (1.42×) | 547 (1.19×) | 475 (1.03×) |
| `lm_head_dgrad` | 1024, 768, 50257 × 1 | 1 | 7182 | 8526 (1.19×) | 13705 (1.91×) | 8526 (1.19×) | 13563 (1.89×) | 8516 (1.19×) | 13538 (1.89×) |
| `lm_head_fwd` | 1024, 50257, 768 × 1 | 1 | 7240 | 8597 (1.19×) | 9403 (1.30×) | 8632 (1.19×) | 9419 (1.30×) | 8598 (1.19×) | 9187 (1.27×) |
| `lm_head_wgrad` | 50257, 768, 1024 × 1 | 1 | 6864 | 8262 (1.20×) | 9490 (1.38×) | 8259 (1.20×) | 9304 (1.36×) | 8187 (1.19×) | 9236 (1.35×) |
| **per training step (ms)** | | | **92.3** | **124.2 (1.35×)** | **140.9 (1.53×)** | **123.8 (1.34×)** | **140.4 (1.52×)** | **115.7 (1.25×)** | **131.8 (1.43×)** |

**GPT-2 medium (H=1024, 16 heads, 24 layers), 1024 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 1 MiB | chunked 1 MiB | staged 2 MiB | chunked 2 MiB | staged 4 MiB | chunked 4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 1024, 1024, 1024 × 1 | 96 | 218 | 268 (1.23×) | 223 (1.02×) | 264 (1.21×) | 223 (1.02×) | 217 (1.00×) | 220 (1.01×) |
| `qkvO_fwd` | 1024, 1024, 1024 × 1 | 96 | 235 | 285 (1.21×) | 231 (0.98×) | 281 (1.20×) | 226 (0.96×) | 233 (0.99×) | 225 (0.96×) |
| `qkvO_wgrad` | 1024, 1024, 1024 × 1 | 96 | 215 | 266 (1.23×) | 220 (1.02×) | 262 (1.22×) | 220 (1.02×) | 214 (1.00×) | 218 (1.01×) |
| `attn_ctx` | 1024, 64, 1024 × 16 | 48 | 255 | 558 (2.18×) | 512 (2.01×) | 553 (2.17×) | 512 (2.00×) | 508 (1.99×) | 499 (1.95×) |
| `attn_dKV` | 1024, 64, 1024 × 16 | 48 | 221 | 521 (2.36×) | 408 (1.85×) | 516 (2.34×) | 408 (1.85×) | 469 (2.13×) | 399 (1.81×) |
| `attn_scores` | 1024, 1024, 64 × 16 | 48 | 295 | 581 (1.97×) | 1182 (4.00×) | 580 (1.96×) | 1184 (4.01×) | 546 (1.85×) | 1178 (3.99×) |
| `mlp_fc_dgrad` | 1024, 4096, 1024 × 1 | 24 | 799 | 945 (1.18×) | 829 (1.04×) | 944 (1.18×) | 828 (1.04×) | 926 (1.16×) | 826 (1.03×) |
| `mlp_fc_fwd` | 1024, 4096, 1024 × 1 | 24 | 821 | 972 (1.18×) | 899 (1.09×) | 964 (1.17×) | 897 (1.09×) | 948 (1.15×) | 840 (1.02×) |
| `mlp_fc_wgrad` | 1024, 4096, 1024 × 1 | 24 | 788 | 935 (1.19×) | 864 (1.10×) | 934 (1.19×) | 864 (1.10×) | 916 (1.16×) | 821 (1.04×) |
| `mlp_proj_dgrad` | 1024, 1024, 4096 × 1 | 24 | 814 | 965 (1.18×) | 841 (1.03×) | 949 (1.16×) | 823 (1.01×) | 934 (1.15×) | 794 (0.97×) |
| `mlp_proj_fwd` | 1024, 1024, 4096 × 1 | 24 | 880 | 1030 (1.17×) | 871 (0.99×) | 1013 (1.15×) | 853 (0.97×) | 999 (1.14×) | 824 (0.94×) |
| `mlp_proj_wgrad` | 4096, 1024, 1024 × 1 | 24 | 788 | 936 (1.19×) | 870 (1.10×) | 935 (1.19×) | 868 (1.10×) | 918 (1.16×) | 824 (1.05×) |
| `lm_head_dgrad` | 1024, 1024, 50257 × 1 | 1 | 9582 | 11104 (1.16×) | 13890 (1.45×) | 11102 (1.16×) | 13649 (1.42×) | 11088 (1.16×) | 13619 (1.42×) |
| `lm_head_fwd` | 1024, 50257, 1024 × 1 | 1 | 9535 | 11080 (1.16×) | 12199 (1.28×) | 11117 (1.17×) | 12272 (1.29×) | 11047 (1.16×) | 11930 (1.25×) |
| `lm_head_wgrad` | 50257, 1024, 1024 × 1 | 1 | 9115 | 10708 (1.17×) | 12712 (1.39×) | 10703 (1.17×) | 12518 (1.37×) | 10627 (1.17×) | 12383 (1.36×) |
| **per training step (ms)** | | | **246.8** | **329.9 (1.34×)** | **328.6 (1.33×)** | **327.4 (1.33×)** | **326.8 (1.32×)** | **305.1 (1.24×)** | **319.5 (1.29×)** |

**GPT-2 large (H=1280, 20 heads, 36 layers), 1024 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 1 MiB | chunked 1 MiB | staged 2 MiB | chunked 2 MiB | staged 4 MiB | chunked 4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 1024, 1280, 1280 × 1 | 144 | 327 | 394 (1.20×) | 528 (1.61×) | 394 (1.20×) | 520 (1.59×) | 394 (1.20×) | 521 (1.59×) |
| `qkvO_fwd` | 1024, 1280, 1280 × 1 | 144 | 353 | 420 (1.19×) | 547 (1.55×) | 418 (1.18×) | 546 (1.55×) | 418 (1.18×) | 544 (1.54×) |
| `qkvO_wgrad` | 1280, 1280, 1024 × 1 | 144 | 316 | 390 (1.23×) | 434 (1.37×) | 383 (1.21×) | 434 (1.37×) | 384 (1.21×) | 434 (1.37×) |
| `attn_ctx` | 1024, 64, 1024 × 20 | 72 | 357 | 720 (2.02×) | 633 (1.77×) | 719 (2.01×) | 633 (1.77×) | 716 (2.00×) | 633 (1.77×) |
| `attn_dKV` | 1024, 64, 1024 × 20 | 72 | 314 | 686 (2.18×) | 504 (1.60×) | 684 (2.18×) | 504 (1.60×) | 684 (2.17×) | 504 (1.60×) |
| `attn_scores` | 1024, 1024, 64 × 20 | 72 | 366 | 721 (1.97×) | 1468 (4.01×) | 719 (1.96×) | 1468 (4.01×) | 719 (1.96×) | 1471 (4.02×) |
| `mlp_fc_dgrad` | 1024, 5120, 1280 × 1 | 36 | 1241 | 1442 (1.16×) | 1284 (1.03×) | 1443 (1.16×) | 1280 (1.03×) | 1442 (1.16×) | 1280 (1.03×) |
| `mlp_fc_fwd` | 1024, 5120, 1280 × 1 | 36 | 1276 | 1479 (1.16×) | 1350 (1.06×) | 1478 (1.16×) | 1352 (1.06×) | 1473 (1.15×) | 1350 (1.06×) |
| `mlp_fc_wgrad` | 1280, 5120, 1024 × 1 | 36 | 1187 | 1394 (1.17×) | 1291 (1.09×) | 1389 (1.17×) | 1292 (1.09×) | 1389 (1.17×) | 1292 (1.09×) |
| `mlp_proj_dgrad` | 1024, 1280, 5120 × 1 | 36 | 1252 | 1449 (1.16×) | 1292 (1.03×) | 1446 (1.15×) | 1287 (1.03×) | 1445 (1.15×) | 1288 (1.03×) |
| `mlp_proj_fwd` | 1024, 1280, 5120 × 1 | 36 | 1342 | 1539 (1.15×) | 1375 (1.02×) | 1541 (1.15×) | 1376 (1.02×) | 1540 (1.15×) | 1375 (1.02×) |
| `mlp_proj_wgrad` | 5120, 1280, 1024 × 1 | 36 | 1188 | 1394 (1.17×) | 1288 (1.08×) | 1389 (1.17×) | 1290 (1.09×) | 1390 (1.17×) | 1291 (1.09×) |
| `lm_head_dgrad` | 1024, 1280, 50257 × 1 | 1 | 11497 | 13229 (1.15×) | 26826 (2.33×) | 13228 (1.15×) | 26815 (2.33×) | 13228 (1.15×) | 26805 (2.33×) |
| `lm_head_fwd` | 1024, 50257, 1280 × 1 | 1 | 11976 | 13772 (1.15×) | 15082 (1.26×) | 13771 (1.15×) | 15079 (1.26×) | 13763 (1.15×) | 15089 (1.26×) |
| `lm_head_wgrad` | 50257, 1280, 1024 × 1 | 1 | 11367 | 13148 (1.16×) | 15659 (1.38×) | 13146 (1.16×) | 15431 (1.36×) | 13146 (1.16×) | 15313 (1.35×) |
| **per training step (ms)** | | | **522.5** | **679.6 (1.30×)** | **746.1 (1.43×)** | **677.6 (1.30×)** | **744.6 (1.42×)** | **677.3 (1.30×)** | **744.3 (1.42×)** |

**GPT-2 xl (H=1600, 25 heads, 48 layers), 1024 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 1 MiB | chunked 1 MiB | staged 2 MiB | chunked 2 MiB | staged 4 MiB | chunked 4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 1024, 1600, 1600 × 1 | 192 | 517 | 606 (1.17×) | 792 (1.53×) | 606 (1.17×) | 791 (1.53×) | 606 (1.17×) | 791 (1.53×) |
| `qkvO_fwd` | 1024, 1600, 1600 × 1 | 192 | 631 | 722 (1.14×) | 795 (1.26×) | 719 (1.14×) | 794 (1.26×) | 720 (1.14×) | 795 (1.26×) |
| `qkvO_wgrad` | 1600, 1600, 1024 × 1 | 192 | 499 | 593 (1.19×) | 768 (1.54×) | 592 (1.19×) | 766 (1.53×) | 592 (1.18×) | 767 (1.54×) |
| `attn_ctx` | 1024, 64, 1024 × 25 | 96 | 471 | 958 (2.04×) | 756 (1.61×) | 961 (2.04×) | 755 (1.60×) | 960 (2.04×) | 756 (1.61×) |
| `attn_dKV` | 1024, 64, 1024 × 25 | 96 | 412 | 875 (2.12×) | 601 (1.46×) | 873 (2.12×) | 600 (1.46×) | 873 (2.12×) | 600 (1.45×) |
| `attn_scores` | 1024, 1024, 64 × 25 | 96 | 455 | 895 (1.97×) | 1827 (4.02×) | 894 (1.97×) | 1831 (4.03×) | 894 (1.97×) | 1834 (4.03×) |
| `mlp_fc_dgrad` | 1024, 6400, 1600 × 1 | 48 | 1862 | 2153 (1.16×) | 2288 (1.23×) | 2144 (1.15×) | 2280 (1.22×) | 2144 (1.15×) | 2279 (1.22×) |
| `mlp_fc_fwd` | 1024, 6400, 1600 × 1 | 48 | 1894 | 2181 (1.15×) | 2378 (1.26×) | 2177 (1.15×) | 2373 (1.25×) | 2176 (1.15×) | 2375 (1.25×) |
| `mlp_fc_wgrad` | 1600, 6400, 1024 × 1 | 48 | 1876 | 2152 (1.15×) | 2466 (1.31×) | 2152 (1.15×) | 2467 (1.32×) | 2152 (1.15×) | 2466 (1.31×) |
| `mlp_proj_dgrad` | 1024, 1600, 6400 × 1 | 48 | 2018 | 2294 (1.14×) | 3045 (1.51×) | 2292 (1.14×) | 3045 (1.51×) | 2292 (1.14×) | 3046 (1.51×) |
| `mlp_proj_fwd` | 1024, 1600, 6400 × 1 | 48 | 2034 | 2320 (1.14×) | 3058 (1.50×) | 2306 (1.13×) | 3058 (1.50×) | 2309 (1.14×) | 3056 (1.50×) |
| `mlp_proj_wgrad` | 6400, 1600, 1024 × 1 | 48 | 1958 | 2238 (1.14×) | 2464 (1.26×) | 2238 (1.14×) | 2464 (1.26×) | 2238 (1.14×) | 2462 (1.26×) |
| `lm_head_dgrad` | 1024, 1600, 50257 × 1 | 1 | 15284 | 17275 (1.13×) | 27726 (1.81×) | 17274 (1.13×) | 27683 (1.81×) | 17274 (1.13×) | 27660 (1.81×) |
| `lm_head_fwd` | 1024, 50257, 1600 × 1 | 1 | 14818 | 16861 (1.14×) | 18612 (1.26×) | 16883 (1.14×) | 18590 (1.25×) | 16855 (1.14×) | 18588 (1.25×) |
| `lm_head_wgrad` | 50257, 1600, 1024 × 1 | 1 | 14373 | 16417 (1.14×) | 20634 (1.44×) | 16364 (1.14×) | 20300 (1.41×) | 16362 (1.14×) | 20140 (1.40×) |
| **per training step (ms)** | | | **1047.9** | **1321.7 (1.26×)** | **1578.4 (1.51×)** | **1319.4 (1.26×)** | **1576.7 (1.50×)** | **1319.4 (1.26×)** | **1576.9 (1.50×)** |

## Per row, 4096 tokens/step


**GPT-2 small (H=768, 12 heads, 12 layers), 4096 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 1 MiB | chunked 1 MiB | staged 2 MiB | chunked 2 MiB | staged 4 MiB | chunked 4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 4096, 768, 768 × 1 | 48 | 461 | 571 (1.24×) | 481 (1.04×) | 569 (1.23×) | 481 (1.04×) | 560 (1.21×) | 534 (1.16×) |
| `qkvO_fwd` | 4096, 768, 768 × 1 | 48 | 468 | 579 (1.24×) | 516 (1.10×) | 576 (1.23×) | 516 (1.10×) | 565 (1.21×) | 553 (1.18×) |
| `qkvO_wgrad` | 768, 768, 4096 × 1 | 48 | 487 | 588 (1.21×) | 815 (1.68×) | 588 (1.21×) | 809 (1.66×) | 580 (1.19×) | 807 (1.66×) |
| `attn_ctx` | 4096, 64, 4096 × 12 | 24 | 2967 | 6228 (2.10×) | 5734 (1.93×) | 6222 (2.10×) | 5734 (1.93×) | 6264 (2.11×) | 5735 (1.93×) |
| `attn_dKV` | 4096, 64, 4096 × 12 | 24 | 2429 | 5709 (2.35×) | 4583 (1.89×) | 5716 (2.35×) | 4584 (1.89×) | 5716 (2.35×) | 4585 (1.89×) |
| `attn_scores` | 4096, 4096, 64 × 12 | 24 | 3410 | 6457 (1.89×) | 15440 (4.53×) | 6459 (1.89×) | 15479 (4.54×) | 6452 (1.89×) | 15426 (4.52×) |
| `mlp_fc_dgrad` | 4096, 3072, 768 × 1 | 12 | 1731 | 2000 (1.16×) | 1867 (1.08×) | 2000 (1.16×) | 1866 (1.08×) | 2000 (1.16×) | 1866 (1.08×) |
| `mlp_fc_fwd` | 4096, 3072, 768 × 1 | 12 | 1835 | 2107 (1.15×) | 2046 (1.11×) | 2106 (1.15×) | 2049 (1.12×) | 2105 (1.15×) | 2046 (1.11×) |
| `mlp_fc_wgrad` | 768, 3072, 4096 × 1 | 12 | 1717 | 1995 (1.16×) | 1753 (1.02×) | 1995 (1.16×) | 1759 (1.02×) | 1995 (1.16×) | 1761 (1.03×) |
| `mlp_proj_dgrad` | 4096, 768, 3072 × 1 | 12 | 1795 | 2079 (1.16×) | 1845 (1.03×) | 2065 (1.15×) | 1849 (1.03×) | 2064 (1.15×) | 1849 (1.03×) |
| `mlp_proj_fwd` | 4096, 768, 3072 × 1 | 12 | 1820 | 2095 (1.15×) | 1877 (1.03×) | 2105 (1.16×) | 1878 (1.03×) | 2105 (1.16×) | 1877 (1.03×) |
| `mlp_proj_wgrad` | 3072, 768, 4096 × 1 | 12 | 1773 | 2050 (1.16×) | 1753 (0.99×) | 2050 (1.16×) | 1755 (0.99×) | 2049 (1.16×) | 1757 (0.99×) |
| `lm_head_dgrad` | 4096, 768, 50257 × 1 | 1 | 28594 | 32429 (1.13×) | 40397 (1.41×) | 32428 (1.13×) | 40227 (1.41×) | 32428 (1.13×) | 40211 (1.41×) |
| `lm_head_fwd` | 4096, 50257, 768 × 1 | 1 | 28504 | 32140 (1.13×) | 37395 (1.31×) | 32201 (1.13×) | 37326 (1.31×) | 32158 (1.13×) | 37309 (1.31×) |
| `lm_head_wgrad` | 50257, 768, 4096 × 1 | 1 | 27049 | 30927 (1.14×) | 35705 (1.32×) | 30928 (1.14×) | 35077 (1.30×) | 30927 (1.14×) | 34775 (1.29×) |
| **per training step (ms)** | | | **491.5** | **768.3 (1.56×)** | **952.4 (1.94×)** | **768.1 (1.56×)** | **952.3 (1.94×)** | **767.6 (1.56×)** | **955.0 (1.94×)** |

**GPT-2 medium (H=1024, 16 heads, 24 layers), 4096 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 1 MiB | chunked 1 MiB | staged 2 MiB | chunked 2 MiB | staged 4 MiB | chunked 4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 4096, 1024, 1024 × 1 | 96 | 798 | 946 (1.18×) | 833 (1.04×) | 945 (1.18×) | 834 (1.04×) | 928 (1.16×) | 911 (1.14×) |
| `qkvO_fwd` | 4096, 1024, 1024 × 1 | 96 | 824 | 971 (1.18×) | 894 (1.09×) | 971 (1.18×) | 894 (1.09×) | 953 (1.16×) | 930 (1.13×) |
| `qkvO_wgrad` | 1024, 1024, 4096 × 1 | 96 | 809 | 948 (1.17×) | 813 (1.00×) | 949 (1.17×) | 815 (1.01×) | 935 (1.16×) | 813 (1.00×) |
| `attn_ctx` | 4096, 64, 4096 × 16 | 48 | 3961 | 8317 (2.10×) | 7590 (1.92×) | 8315 (2.10×) | 7588 (1.92×) | 8322 (2.10×) | 7588 (1.92×) |
| `attn_dKV` | 4096, 64, 4096 × 16 | 48 | 3195 | 7575 (2.37×) | 6066 (1.90×) | 7575 (2.37×) | 6068 (1.90×) | 7575 (2.37×) | 6067 (1.90×) |
| `attn_scores` | 4096, 4096, 64 × 16 | 48 | 4546 | 8604 (1.89×) | 20612 (4.53×) | 8605 (1.89×) | 20652 (4.54×) | 8606 (1.89×) | 20578 (4.53×) |
| `mlp_fc_dgrad` | 4096, 4096, 1024 × 1 | 24 | 3097 | 3467 (1.12×) | 3264 (1.05×) | 3467 (1.12×) | 3265 (1.05×) | 3467 (1.12×) | 3264 (1.05×) |
| `mlp_fc_fwd` | 4096, 4096, 1024 × 1 | 24 | 3229 | 3601 (1.11×) | 3559 (1.10×) | 3601 (1.12×) | 3557 (1.10×) | 3601 (1.12×) | 3556 (1.10×) |
| `mlp_fc_wgrad` | 1024, 4096, 4096 × 1 | 24 | 3098 | 3487 (1.13×) | 3101 (1.00×) | 3487 (1.13×) | 3104 (1.00×) | 3486 (1.13×) | 3122 (1.01×) |
| `mlp_proj_dgrad` | 4096, 1024, 4096 × 1 | 24 | 3128 | 3517 (1.12×) | 3158 (1.01×) | 3518 (1.12×) | 3161 (1.01×) | 3518 (1.12×) | 3168 (1.01×) |
| `mlp_proj_fwd` | 4096, 1024, 4096 × 1 | 24 | 3210 | 3601 (1.12×) | 3207 (1.00×) | 3594 (1.12×) | 3206 (1.00×) | 3594 (1.12×) | 3209 (1.00×) |
| `mlp_proj_wgrad` | 4096, 1024, 4096 × 1 | 24 | 3098 | 3488 (1.13×) | 3100 (1.00×) | 3488 (1.13×) | 3117 (1.01×) | 3488 (1.13×) | 3126 (1.01×) |
| `lm_head_dgrad` | 4096, 1024, 50257 × 1 | 1 | 38118 | 42141 (1.11×) | 53833 (1.41×) | 42141 (1.11×) | 53644 (1.41×) | 42142 (1.11×) | 53563 (1.41×) |
| `lm_head_fwd` | 4096, 50257, 1024 × 1 | 1 | 37623 | 41450 (1.10×) | 48400 (1.29×) | 41454 (1.10×) | 48437 (1.29×) | 41469 (1.10×) | 48372 (1.29×) |
| `lm_head_wgrad` | 50257, 1024, 4096 × 1 | 1 | 36319 | 40395 (1.11×) | 47945 (1.32×) | 40395 (1.11×) | 47098 (1.30×) | 40395 (1.11×) | 46695 (1.29×) |
| **per training step (ms)** | | | **1359.8** | **2082.6 (1.53×)** | **2504.2 (1.84×)** | **2082.5 (1.53×)** | **2505.9 (1.84×)** | **2078.1 (1.53×)** | **2513.3 (1.85×)** |

**GPT-2 large (H=1280, 20 heads, 36 layers), 4096 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 1 MiB | chunked 1 MiB | staged 2 MiB | chunked 2 MiB | staged 4 MiB | chunked 4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 4096, 1280, 1280 × 1 | 144 | 1202 | 1389 (1.16×) | 1290 (1.07×) | 1390 (1.16×) | 1288 (1.07×) | 1390 (1.16×) | 1289 (1.07×) |
| `qkvO_fwd` | 4096, 1280, 1280 × 1 | 144 | 1280 | 1468 (1.15×) | 1367 (1.07×) | 1468 (1.15×) | 1367 (1.07×) | 1468 (1.15×) | 1366 (1.07×) |
| `qkvO_wgrad` | 1280, 1280, 4096 × 1 | 144 | 1206 | 1389 (1.15×) | 1613 (1.34×) | 1389 (1.15×) | 1613 (1.34×) | 1389 (1.15×) | 1613 (1.34×) |
| `attn_ctx` | 4096, 64, 4096 × 20 | 72 | 4964 | 10452 (2.11×) | 9132 (1.84×) | 10437 (2.10×) | 9130 (1.84×) | 10432 (2.10×) | 9132 (1.84×) |
| `attn_dKV` | 4096, 64, 4096 × 20 | 72 | 3983 | 9456 (2.37×) | 7279 (1.83×) | 9456 (2.37×) | 7277 (1.83×) | 9456 (2.37×) | 7276 (1.83×) |
| `attn_scores` | 4096, 4096, 64 × 20 | 72 | 5680 | 10750 (1.89×) | 25774 (4.54×) | 10751 (1.89×) | 25830 (4.55×) | 10751 (1.89×) | 25729 (4.53×) |
| `mlp_fc_dgrad` | 4096, 5120, 1280 × 1 | 36 | 4690 | 5177 (1.10×) | 4846 (1.03×) | 5176 (1.10×) | 4846 (1.03×) | 5176 (1.10×) | 4845 (1.03×) |
| `mlp_fc_fwd` | 4096, 5120, 1280 × 1 | 36 | 4826 | 5321 (1.10×) | 5157 (1.07×) | 5326 (1.10×) | 5152 (1.07×) | 5326 (1.10×) | 5151 (1.07×) |
| `mlp_fc_wgrad` | 1280, 5120, 4096 × 1 | 36 | 4721 | 5219 (1.11×) | 4825 (1.02×) | 5221 (1.11×) | 4828 (1.02×) | 5222 (1.11×) | 4830 (1.02×) |
| `mlp_proj_dgrad` | 4096, 1280, 5120 × 1 | 36 | 4684 | 5185 (1.11×) | 4936 (1.05×) | 5185 (1.11×) | 4933 (1.05×) | 5185 (1.11×) | 4939 (1.05×) |
| `mlp_proj_fwd` | 4096, 1280, 5120 × 1 | 36 | 5062 | 5573 (1.10×) | 4944 (0.98×) | 5560 (1.10×) | 4943 (0.98×) | 5562 (1.10×) | 4946 (0.98×) |
| `mlp_proj_wgrad` | 5120, 1280, 4096 × 1 | 36 | 4722 | 5221 (1.11×) | 4825 (1.02×) | 5224 (1.11×) | 4828 (1.02×) | 5224 (1.11×) | 4829 (1.02×) |
| `lm_head_dgrad` | 4096, 1280, 50257 × 1 | 1 | 47643 | 51853 (1.09×) | 67279 (1.41×) | 51853 (1.09×) | 67066 (1.41×) | 51852 (1.09×) | 66946 (1.41×) |
| `lm_head_fwd` | 4096, 50257, 1280 × 1 | 1 | 47273 | 51284 (1.08×) | 60125 (1.27×) | 51233 (1.08×) | 60074 (1.27×) | 51214 (1.08×) | 60084 (1.27×) |
| `lm_head_wgrad` | 50257, 1280, 4096 × 1 | 1 | 44860 | 49122 (1.10×) | 59183 (1.32×) | 49141 (1.10×) | 58128 (1.30×) | 49160 (1.10×) | 57669 (1.29×) |
| **per training step (ms)** | | | **2757.4** | **4112.1 (1.49×)** | **4901.9 (1.78×)** | **4111.1 (1.49×)** | **4904.0 (1.78×)** | **4110.8 (1.49×)** | **4896.7 (1.78×)** |

**GPT-2 xl (H=1600, 25 heads, 48 layers), 4096 tokens/step**

| GEMM row | m, n, k × batch | calls | cuBLAS µs | staged 1 MiB | chunked 1 MiB | staged 2 MiB | chunked 2 MiB | staged 4 MiB | chunked 4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qkvO_dgrad` | 4096, 1600, 1600 × 1 | 192 | 1958 | 2196 (1.12×) | 2690 (1.37×) | 2196 (1.12×) | 2691 (1.37×) | 2196 (1.12×) | 2693 (1.38×) |
| `qkvO_fwd` | 4096, 1600, 1600 × 1 | 192 | 2096 | 2333 (1.11×) | 2754 (1.31×) | 2331 (1.11×) | 2753 (1.31×) | 2326 (1.11×) | 2752 (1.31×) |
| `qkvO_wgrad` | 1600, 1600, 4096 × 1 | 192 | 1889 | 2126 (1.13×) | 2899 (1.54×) | 2121 (1.12×) | 2904 (1.54×) | 2122 (1.12×) | 2901 (1.54×) |
| `attn_ctx` | 4096, 64, 4096 × 25 | 96 | 6388 | 13244 (2.07×) | 11444 (1.79×) | 13265 (2.08×) | 11448 (1.79×) | 13273 (2.08×) | 11449 (1.79×) |
| `attn_dKV` | 4096, 64, 4096 × 25 | 96 | 4776 | 11618 (2.43×) | 9144 (1.91×) | 11619 (2.43×) | 9143 (1.91×) | 11618 (2.43×) | 9142 (1.91×) |
| `attn_scores` | 4096, 4096, 64 × 25 | 96 | 7098 | 13434 (1.89×) | 32214 (4.54×) | 13434 (1.89×) | 32213 (4.54×) | 13435 (1.89×) | 32114 (4.52×) |
| `mlp_fc_dgrad` | 4096, 6400, 1600 × 1 | 48 | 7394 | 8028 (1.09×) | 9165 (1.24×) | 8028 (1.09×) | 9132 (1.24×) | 8028 (1.09×) | 9120 (1.23×) |
| `mlp_fc_fwd` | 4096, 6400, 1600 × 1 | 48 | 7537 | 8200 (1.09×) | 9564 (1.27×) | 8173 (1.08×) | 9564 (1.27×) | 8163 (1.08×) | 9568 (1.27×) |
| `mlp_fc_wgrad` | 1600, 6400, 4096 × 1 | 48 | 7234 | 7890 (1.09×) | 9313 (1.29×) | 7890 (1.09×) | 9311 (1.29×) | 7890 (1.09×) | 9308 (1.29×) |
| `mlp_proj_dgrad` | 4096, 1600, 6400 × 1 | 48 | 7688 | 8338 (1.08×) | 10335 (1.34×) | 8339 (1.08×) | 10331 (1.34×) | 8338 (1.08×) | 10326 (1.34×) |
| `mlp_proj_fwd` | 4096, 1600, 6400 × 1 | 48 | 8102 | 8752 (1.08×) | 10592 (1.31×) | 8769 (1.08×) | 10593 (1.31×) | 8751 (1.08×) | 10588 (1.31×) |
| `mlp_proj_wgrad` | 6400, 1600, 4096 × 1 | 48 | 7734 | 8392 (1.08×) | 9252 (1.20×) | 8392 (1.08×) | 9258 (1.20×) | 8391 (1.08×) | 9251 (1.20×) |
| `lm_head_dgrad` | 4096, 1600, 50257 × 1 | 1 | 61962 | 66399 (1.07×) | 94525 (1.53×) | 66400 (1.07×) | 94420 (1.52×) | 66401 (1.07×) | 94344 (1.52×) |
| `lm_head_fwd` | 4096, 50257, 1600 × 1 | 1 | 58565 | 62984 (1.08×) | 73815 (1.26×) | 62923 (1.07×) | 73785 (1.26×) | 62868 (1.07×) | 73796 (1.26×) |
| `lm_head_wgrad` | 50257, 1600, 4096 × 1 | 1 | 57029 | 61542 (1.08×) | 78115 (1.37×) | 61549 (1.08×) | 76696 (1.34×) | 61568 (1.08×) | 76030 (1.33×) |
| **per training step (ms)** | | | **5264.7** | **7525.7 (1.43×)** | **9712.1 (1.84×)** | **7526.0 (1.43×)** | **9709.9 (1.84×)** | **7524.8 (1.43×)** | **9698.0 (1.84×)** |

## Reading the numbers

- **Staging costs 1.24–1.34× cuBLAS per step at 1024 tokens and
  1.43–1.56× at 4096 tokens**, whatever the class. On the attention rows
  it is about 2× (the per-head score matrices are whole chunks that must
  be gathered and scattered), on the MLP, projection and lm-head rows
  1.07–1.24×, except the projection rows whose every operand is a single
  chunk on the 4 MiB class, where staging is plain cuBLAS.
- **The class matters to staging only when an operand stops being
  chunked.** The 1 MiB and 2 MiB classes give the same picture. At 4 MiB
  the hidden state and the fc weight of GPT-2 small and medium at 1024
  tokens fit one chunk, so fewer bytes are copied: staged drops from 1.34×
  to 1.25× of cuBLAS per step for small and from 1.33× to 1.24× for
  medium, and on their projection rows staging is plain cuBLAS. The
  chunked route shows the same step: 1.24× to 1.02× for small.
- **Over a whole step the chunked route is slower than staging on every
  model.** It takes 1.30–1.52× cuBLAS per step at 1024 tokens and
  1.78–1.95× at 4096 tokens. By family (rows counted once per chunk
  class, 360 in all):

| family | rows with a chunked route | of those, faster than staged | chunked / cuBLAS, median (range) |
|---|---:|---:|---|
| MLP | 144 of 144 | 100 | 1.06× (0.94–1.51×) |
| projection | 72 of 72 | 30 | 1.26× (0.96–1.68×) |
| attention | 72 of 72 | 48 | 1.91× (1.43–4.55×) |
| lm head | 72 of 72 | 0 | 1.35× (1.25–2.33×) |

  Over all 360 the chunked route is faster than cuBLAS on 16, within 2 %
  on 35, within 10 % on 119, and faster than staging on 178.
- **The MLP GEMMs of GPT-2 small, medium and large are the one family
  where the chunked route competes.** Of their 36 rows it is faster than
  staged on 32 at 1 MiB, 32 at 2 MiB and all 36 at 4 MiB; summed over the
  six rows of a step it stands at 1.01–1.05× cuBLAS for medium and large
  and at 4096 tokens for small too, against 1.10–1.18× for staging. Per
  row it is within 2 % of cuBLAS on 8, 8 and 9 of the 36, within 10 % on
  29, 29 and 34. The worst rows are GPT-2 small at 1024 tokens on the
  1 MiB and 2 MiB classes (1.27–1.42×) and `mlp_fc_fwd` at 4096 tokens
  (1.10–1.12×).
- **GPT-2 XL and the lm head now have chunked numbers, and they lose to
  staging everywhere.** The `tails` kernel is one general tile shape
  with predicated loads, not a tuned one: on XL it takes 1.20–1.51× cuBLAS
  on the MLP rows and 1.26–1.54× on the projection rows (staging 1.08–1.19×);
  on the lm-head rows of all models 1.25–2.33× (staging 1.07–1.20×). The
  worst is `lm_head_dgrad` at 1024 tokens, 1.42–2.33×: a 1024 x 1600
  output with k = 50257 is 104 threadblocks of 6283 k-panels, work that
  needs a split along k.
- **The projection rows favor the chunked route only for GPT-2 medium**,
  where it is at 1.00–1.09× cuBLAS against 1.16–1.21× for staging. For
  small and large it is behind: `qkvO_wgrad` at 1.6–1.7× for small and
  1.3–1.4× for large (staging 1.2×), and large's `qkvO_fwd` and
  `qkvO_dgrad` at 1024 tokens at 1.5–1.6×.
- **Attention is lost by both routes, the chunked one further.** The
  three batched attention rows cost staging 1.8–2.1× cuBLAS per step and
  the chunked route 2.4–2.9×.

**What this says.** With every row of the step on the native family, the
chunked route costs a GPT-2 step 1.3–1.5× contiguous cuBLAS at 1024
tokens and 1.8–1.95× at 4096 tokens, where staging costs 1.24–1.56×. It
is not an improvement over the route it would replace on any model. The
MLP GEMMs of GPT-2 small, medium and large are the one family where it
does (1.01–1.05× cuBLAS in aggregate, faster than staging on 100 of 108
rows); attention, XL and the lm head are lost. Production routing is
unchanged: the wrapper still sends these calls to plain cuBLAS or to
staging. Where the time goes is in `gemm_chunked_status.md`.

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
