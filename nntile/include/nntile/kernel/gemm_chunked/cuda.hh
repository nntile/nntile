/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/gemm_chunked/cuda.hh
 * Native chunked fp32 GEMM kernels: C-order row-major
 * C = alpha*op(A)*op(B) + beta*C over chunk-table operands (any
 * compile-time power-of-two class >= 1 MiB)
 * (no workspace staging), with a shape-driven implementation picker.
 *
 * @version 1.0.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

#ifdef NNTILE_USE_CUDA

// Standard libraries
#include <cstddef>

// Third-party headers
#include <cuda_runtime.h>

namespace nntile::kernel::gemm_chunked
{

//! Portability contract: this family is PURE CUDA — no tensor ops, no
//! cp.async, no arch intrinsics — and every configuration keeps shared
//! memory at or below 32 KiB, so the kernels build and run on every
//! pre-Ampere architecture (sm_50 Maxwell through sm_75 Turing) as
//! well as later ones. The chunk CLASS is a runtime shift: any
//! compile-time power-of-two class of at least 1 MiB (the engine's
//! NNHAUL_CHUNK_SIZE_MB law) is supported — the kernel test sweeps
//! 1/2/4/8 MiB. Pre-Ampere is the family's raison d'etre: those
//! GPUs have no fp32-input tensor path (TF32 is sm_80+), and no library
//! consumes CHUNK-TABLE operands — the feature this family exists for.
//! (Correction of an earlier claim: CUTLASS 2.11's generic SIMT fp32
//! path does run on sm_75, measured at 0.97-0.99x of cuBLAS on flat
//! operands — an earlier 'no such kernel / it crashes' verdict was an
//! artifact of two bugs in the campaign's side benchmark. CUTLASS's
//! mainloop is the adopted scheduling reference for future rounds.) Tuning is measured on sm_75 (RTX 2080 Ti, SM clock
//! pinned at the 1350 MHz base for reproducibility); per-arch tile
//! retunes are follow-ups.

//! Operand addressing for the chunk-native kernels.
//!
//! A multi-chunk tile is the LOGICAL CONCATENATION of its whole-class
//! chunks; the kernel resolves any element `e` (within one operand's
//! linear space, batch offset included) as
//! `tab[e >> shift] + (e & ((1 << shift) - 1))`. Three addressing modes:
//!
//!   - flat:    `tab` is really a `const float*` base and `shift == 0`
//!              (element offset added directly, no indirection);
//!   - table:   `tab` is a device `const float* const*` (a chunk-ptr
//!              table view) and `shift == log2(chunk floats)` for the
//!              compile-time chunk class;
//!   - one-shot:`tab` is a 1-entry device table and `shift == 62`, the
//!              single-chunk sentinel (element >> 62 == 0 for every
//!              representable element, so the table path degenerates
//!              to tab[0] + e).
//!
//! The producer of C passes a `float* const*` (mutable chunk bases) in
//! the same encoding.
struct Operand
{
    const void *ptr = nullptr; //!< flat base or device chunk table
    int shift = 0;             //!< 0 flat, 62 sentinel, else log2 chunk
};

//! Ownership law (FINAL, owner directive): the native family owns a
//! call only when every shape dimension except batch is exactly 64 or
//! a multiple of 128. All GPT-2 small/medium/large GEMMs conform (768,
//! 1024, 1280 and their 3H/4H multiples; attention projections are
//! exact 64s); lm_head's 50257 vocab and GPT-2 XL's 1600 (= 12.5x128)
//! belong to the workspace-staged cuBLAS path, as does anything ragged.
//! Owned calls carry no shape tails, row starts are at least 64-float
//! aligned, and — chunk classes being >= 2^18 floats — 16-float-aligned
//! minichunks never straddle chunks, so loaders resolve pointers only
//! at chunk-boundary crossings of a row march. fp32 only (the campaign
//! dtype); every other repr type takes the staged path.
//!
//! Transposition flags in C-order row-major semantics.
enum class Op : int
{
    NoTrans = 0,
    Trans = 1
};

//! True when a native kernel covers the call (shape family + alignment
//! laws of the vectorized paths). False means the caller must fall back
//! to the existing workspace-staged cuBLAS path. Never throws.
bool supports(Op transA, Op transB, int m, int n, int k, int batch) noexcept;

//! Human-readable name of the implementation the picker would choose
//! (diagnostics and the benchmark harness; empty string when
//! !supports()).
const char *pick_name(Op transA, Op transB, int m, int n, int k,
    int batch) noexcept;

//! Launch the best native configuration for the shape REGARDLESS of
//! the ownership policy — the workspace-overflow rescue (a staged call
//! whose bytes exceed the gemm workspace has no other path). Same
//! contract as launch() otherwise; returns false when unsupported.
bool launch_forced(Op transA, Op transB, int m, int n, int k, int batch,
    float alpha, const Operand &a, const Operand &b, float beta,
    const Operand &c, long long stride_a, long long stride_b,
    long long stride_c, cudaStream_t stream) noexcept;

//! Launch the picked native kernel on `stream`. The operand strides are
//! in ELEMENTS per batch index (0 = broadcast, matching the cuBLAS
//! wrapper's stride semantics). `alpha`/`beta` are host scalars.
//!
//! Returns false without launching when !supports(...) — the caller's
//! fallback owns the call. Device tables must stay alive until the
//! stream retires the launch (the chunk-ptr table discipline).
//!
//! The result is fp32 FMA-accumulated (not bitwise cuBLAS); the
//! chunked-tiles accuracy checks bound it against a host reference.
bool launch(Op transA, Op transB, int m, int n, int k, int batch,
    float alpha, const Operand &a, const Operand &b, float beta,
    const Operand &c, long long stride_a, long long stride_b,
    long long stride_c, cudaStream_t stream) noexcept;

//! [campaign2/p6-staged] FLAT-ENGINE ENTRY for the workspace-staged
//! path's GEMM leg: run the p5-cutlass per-shape route table's CUTLASS
//! engine on FLAT device pointers (a staged workspace slot or a
//! contiguous leading chunk) for one BATCH-1, BETA-0, classic-shape
//! call -- WINNERS ONLY (the row's measured p5 flat ratio at or under
//! 0.995; the losing x1.00-x1.04 rows keep cuBLAS so the staged bar
//! never regresses). Table-only and side-effect-free on decline: no
//! pin/env consultation, no native-id routing, and without the CUTLASS
//! headers on the include path it is a constant false (the staged leg
//! keeps its cuBLAS call byte-for-byte).
//!
//! Returns true when the engine ran (C holds alpha*op(A)*op(B), every
//! element overwritten, C never read); false when NOTHING launched and
//! the caller's cuBLAS fallback owns the call verbatim (beta != 0,
//! foreign shape, null pointers, engine-less build).
bool flat_engine(Op transA, Op transB, int m, int n, int k, float alpha,
    const void *a, const void *b, float beta, void *c,
    cudaStream_t stream) noexcept;

} // namespace nntile::kernel::gemm_chunked

#endif // NNTILE_USE_CUDA
