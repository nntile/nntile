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
 * C = alpha*op(A)*op(B) + beta*C over 4 MiB-class chunk-table operands
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
//! well as later ones. Pre-Ampere is the family's raison d'etre: those
//! GPUs have no fp32-input tensor path (TF32 is sm_80+), CUTLASS ships
//! no fp32-SIMT GEMM for them, and Triton's backend is not maintained
//! below sm_80 — a hand-tuned FFMA kernel is the only native-chunked
//! option there. Tuning is measured on sm_75 (RTX 2080 Ti, SM clock
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
//!              ring view) and `shift == log2(chunk floats)` for the
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
//! stream retires the launch (the chunk-ptr ring discipline).
//!
//! The result is fp32 FMA-accumulated (not bitwise cuBLAS); the
//! chunked-tiles accuracy checks bound it against a host reference.
bool launch(Op transA, Op transB, int m, int n, int k, int batch,
    float alpha, const Operand &a, const Operand &b, float beta,
    const Operand &c, long long stride_a, long long stride_b,
    long long stride_c, cudaStream_t stream) noexcept;

} // namespace nntile::kernel::gemm_chunked

#endif // NNTILE_USE_CUDA
