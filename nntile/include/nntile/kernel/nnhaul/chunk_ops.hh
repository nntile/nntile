/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/nnhaul/chunk_ops.hh
 * Chunk-addressed structured kernels over NNHaul chunk tables.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/defs.h>

#if defined(NNTILE_USE_CUDA) && defined(NNTILE_USE_NNHAUL)

#include <nnhaul/nnhaul.hh>

#include <cstddef>

namespace nntile::kernel::nnhaul
{

/*! Chunk-addressed flat elementwise multiply (in-codelet only):
 * dst[p] = alpha * src[p] * dst[p] for p in [0, nelems). ONE launch;
 * the chunk class is the compile-time kChunkClassBytes, so the flat
 * position maps to (chunk, offset) with a shift/mask. */
void chunk_multiply_inplace(::nnhaul::Buffer const &src,
    ::nnhaul::Buffer &dst, std::size_t nelems, float alpha,
    ::nnhaul::ElemType type);

/*! Chunk-addressed flat elementwise SGD-with-momentum step (in-codelet
 * only): the exact kernel::sgd_step arithmetic over num_elems flat
 * positions. ONE launch. */
void chunk_sgd_step(::nnhaul::Buffer const &grad,
    ::nnhaul::Buffer &velocity, ::nnhaul::Buffer &param, long long num_iter,
    std::size_t num_elems, float momentum, float lr, float weight_decay,
    float dampening, bool nesterov, ::nnhaul::ElemType type);

/*! Chunk-addressed subtract_indexed_outputs (in-codelet only):
 * dst[labels[i], i] -= val for each output column i (Fortran layout,
 * column i is the contiguous n_labels-element run at i*n_labels); an
 * ignore_index column is zeroed. ONE launch. */
void chunk_subtract_indexed_outputs(::nnhaul::Buffer const &labels,
    ::nnhaul::Buffer &dst, std::size_t n_labels, std::size_t n_outputs,
    long long ignore_index, float val, ::nnhaul::ElemType type);

/*! Chunk-addressed sum_fiber (in-codelet only):
 * dst[i2 + k*b] = beta*dst[i2 + k*b] + alpha * sum_{i0,i1}
 * src[i0 + m*(i2 + k*(i1 + n*b))]. ONE launch, same thread geometry
 * and per-thread accumulation order as kernel::sum_fiber::cuda. */
void chunk_sum_fiber(::nnhaul::Buffer const &src, ::nnhaul::Buffer &dst,
    std::size_t m, std::size_t n, std::size_t k, std::size_t batch,
    float alpha, float beta, ::nnhaul::ElemType type);

/*! Chunk-addressed total_sum_accum (in-codelet only):
 * val += alpha*(logsumexp[i] - src[labels[i] + i*n_labels]) over all
 * output columns i. ONE launch. */
void chunk_total_sum_accum(::nnhaul::Buffer const &logsumexp,
    ::nnhaul::Buffer const &src, ::nnhaul::Buffer const &labels,
    ::nnhaul::Buffer &val, float alpha, std::size_t n_labels,
    std::size_t n_outputs, long long ignore_index, ::nnhaul::ElemType type);

/*! Chunk-addressed maxsumexp (in-codelet only): the (max, sumexp) pair
 * of every (i0, i2) unit over the k middle positions. ONE launch, the
 * same thread mapping and per-thread reduction order as
 * kernel::maxsumexp::cuda (both the m==1 and m>1 paths). */
void chunk_maxsumexp(::nnhaul::Buffer const &src, ::nnhaul::Buffer &dst,
    std::size_t m, std::size_t n, std::size_t k, float beta,
    ::nnhaul::ElemType type);

/*! Chunk-addressed swap of axes 1 and 3 of the 5-D view
 * [d0, d1, d2, d3, d4] -> [d0, d3, d2, d1, d4] (in-codelet only).
 * The kernel reads a destination flat position, decodes it with the
 * destination axes, and writes the corresponding source position; both
 * sides go through their chunk tables with the compile-time shift/mask
 * address map. ONE launch when the chunk tables fit the by-value
 * ceiling (128 chunks); wider operands are split into contiguous dst
 * ranges whose source chunk span fits (never whole-buffer staging). */
void chunk_swap_two_axes(::nnhaul::Buffer const &src, ::nnhaul::Buffer &dst,
    std::size_t d0, std::size_t d1, std::size_t d2, std::size_t d3,
    std::size_t d4, ::nnhaul::ElemType type);

/*! Chunk-addressed transpose (in-codelet only): the exact arithmetic of
 * kernel::transpose (dst[i*n+j] = alpha * src[i+j*m]) through
 * chunk tables, in a shared-memory tile that coalesces the destination
 * writes and the source reads. ONE launch when the chunk tables fit
 * the by-value ceiling (128 chunks); wider operands are split over the
 * destination tile grid into pieces whose source/destination chunk
 * spans fit (never whole-buffer staging). */
void chunk_transpose(::nnhaul::Buffer const &src, ::nnhaul::Buffer &dst,
    std::size_t m, std::size_t n, float alpha, ::nnhaul::ElemType type);

/*! Chunk-addressed forward conv2d (in-codelet only): the exact
 * arithmetic of kernel::conv2d_inplace (Kahan accumulation, same
 * traversal order) with every global load/store mapped through the
 * operand's chunk table. ONE launch when the three chunk spans fit the
 * by-value ceiling (128 chunks each); wider operands are split into
 * contiguous destination ranges whose source spans fit (never
 * whole-buffer staging). Negative padding offsets are supported. */
void chunk_conv2d_inplace(::nnhaul::Buffer const &src1,
    ::nnhaul::Buffer const &src2, ::nnhaul::Buffer &dst, long long src1_m,
    long long src1_n, long long src1_channels, long long batch,
    long long src2_m, long long src2_n, long long dilation_m,
    long long dilation_n, long long dst_channels, long long offset_m,
    long long offset_n, float alpha, long long dst_m, long long dst_n,
    long long stride_m, long long stride_n, float beta,
    ::nnhaul::ElemType type);

/*! Chunk-addressed conv2d backward-to-input (in-codelet only): the
 * exact arithmetic of kernel::conv2d_bwd_input_inplace through chunk
 * tables, with the same split-over-destination-ranges fallback. */
void chunk_conv2d_bwd_input_inplace(::nnhaul::Buffer const &src1,
    ::nnhaul::Buffer const &src2, ::nnhaul::Buffer &dst, long long src1_m,
    long long src1_n, long long stride_m, long long stride_n,
    long long src1_channels, long long batch, long long src2_m,
    long long src2_n, long long dilation_m, long long dilation_n,
    long long dst_channels, long long offset_m, long long offset_n,
    float alpha, long long dst_m, long long dst_n, float beta,
    ::nnhaul::ElemType type);

/*! Chunk-addressed conv2d backward-to-weight (in-codelet only): the
 * exact arithmetic of kernel::conv2d_bwd_weight_inplace through chunk
 * tables, with the same split-over-destination-ranges fallback. */
void chunk_conv2d_bwd_weight_inplace(::nnhaul::Buffer const &src1,
    ::nnhaul::Buffer const &src2, ::nnhaul::Buffer &dst, long long src1_m,
    long long src1_n, long long src1_channels, long long batch,
    long long src2_m, long long src2_n, long long stride_m,
    long long stride_n, long long src2_channels, long long offset_m,
    long long offset_n, float alpha, long long dst_m, long long dst_n,
    long long dilation_m, long long dilation_n, float beta,
    ::nnhaul::ElemType type);

/*! Chunk-addressed randn (in-codelet only): fills the output exactly as
 * kernel::randn::cuda does, with the per-position chameleon value
 * (seed jumped by the absolute underlying position) and the strided
 * output offset mapped through the chunk table. ONE launch when the
 * output's touched chunk span fits the by-value ceiling; wider outputs
 * are split over contiguous flat position ranges (never staging). */
void chunk_randn(::nnhaul::Buffer &data, std::size_t nelems,
    unsigned long long seed, float mean, float stddev, long long ndim,
    long long const *start, long long const *shape,
    long long const *stride, long long const *underlying_shape,
    ::nnhaul::ElemType type);

} // namespace nntile::kernel::nnhaul

#endif // NNTILE_USE_CUDA && NNTILE_USE_NNHAUL
