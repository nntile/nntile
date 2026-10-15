/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/conv2d_inplace.cc
 * Forward 2D-Convolution of two tensors in WHCN format
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/conv2d_inplace.hh"
#include "nntile/nnhaul/stage.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/conv2d_inplace.hh"

namespace nntile::haul
{
// ---------------------------------------------------------------------------
// DELIBERATELY STAGED (the family-3 derivation, docs/chunked-tiles.md
// §4.3): conv2d_inplace is NOT cleanly chunk-decomposable with the
// existing kernel signature.
//
// Index map (kernel::conv2d_inplace::cpu/cuda): all three tensors are
// F-contiguous WHCN, so the contiguous axis is m (axis 0) on src1, src2
// and dst alike. dst(i,j,k,b) sums src1(m,n,l,b) over the m-window
// [i*stride_m - offset_m, i*stride_m - offset_m + dilation_m*(src2_m-1)]
// and the analogous n-window. A sub-call over a dst m-run [i0, i0+Li)
// IS kernel-expressible (dst_m'=Li at dst+i0, offset_m'=offset_m -
// i0*stride_m, src1/src2 args unchanged - the kernel's own bounds logic
// then skips out-of-window contributions exactly as the plain call
// does for those outputs), and the pieces would beta-chain along the
// m-runs like the fiber class. What blocks native promotion is the
// piece CAP: the src1 m-read of one piece is the CONTIGUOUS range
// [i0*stride_m - offset_m, (i0+Li-1)*stride_m - offset_m +
// dilation_m*(src2_m-1)] taken at EVERY (n,l,b) column of src1, whose
// flat starts are src1_m apart - each column lands on a different
// residue mod chunk_elems, so the whole-range-in-one-chunk cap is the
// MINIMUM over n*channels*batch columns and collapses Li to ~1 for
// small chunks; straddling windows would additionally need per-column
// staging of a window that the scatter must write back before the NEXT
// piece's overlapping window reads it (windows of adjacent dst rows
// OVERLAP, so staged windows cannot retire independently). GPT-2 does
// not use conv2d; the universal staging fallback is exact. Revisit only
// with a panel-style kernel variant (kernel::transpose::panel's
// whole-tensor-strides trick) that takes per-column base pointers.
// ---------------------------------------------------------------------------

//! Constructor
template<typename T>
Conv2dInplace<std::tuple<T>>::Conv2dInplace():
    codelet(
        "nntile_conv2d_inplace",
        &Conv2dInplace<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Conv2dInplace<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Conv2dInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply conv2d_inplace on StarPU buffer on CPU
template<typename T>
void Conv2dInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    const T *src1 = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *src2 = ::nntile::haul::buf_as<T>(buffers, 1);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 2);
    // Launch kernel
    kernel::conv2d_inplace::cpu<T>(
        args->src1_m,
        args->src1_n,
        args->src1_channels,
        args->batch,
        args->src2_m,
        args->src2_n,
        args->dilation_m,
        args->dilation_n,
        args->dst_channels,
        args->offset_m,
        args->offset_n,
        args->alpha,
        src1,
        src2,
        args->dst_m,
        args->dst_n,
        args->stride_m,
        args->stride_n,
        args->beta,
        dst
    );
}

#ifdef NNTILE_USE_CUDA
//! Apply conv2d_inplace on StarPU buffer on CUDA
template<typename T>
void Conv2dInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src1 = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *src2 = ::nntile::haul::buf_as<T>(buffers, 1);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 2);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Launch kernel
    kernel::conv2d_inplace::cuda<T>(
        stream,
        args->src1_m,
        args->src1_n,
        args->src1_channels,
        args->batch,
        args->src2_m,
        args->src2_n,
        args->dilation_m,
        args->dilation_n,
        args->dst_channels,
        args->offset_m,
        args->offset_n,
        args->alpha,
        src1,
        src2,
        args->dst_m,
        args->dst_n,
        args->stride_m,
        args->stride_n,
        args->beta,
        dst
    );
}
#endif // NNTILE_USE_CUDA

//! Footprint for conv2d_inplace tasks
template<typename T>
std::uint64_t Conv2dInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over all args
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(args, sizeof(*args), hash);
    return hash;
}

//! Submit conv2d_inplace task
template<typename T>
void Conv2dInplace<std::tuple<T>>::submit(int starpu_worker_hint, Index src1_m, Index src1_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index dilation_m, Index dilation_n,
        Index dst_channels, Index offset_m, Index offset_n, Scalar alpha,
        ::nnhaul::Handle & src1, ::nnhaul::Handle & src2, Index dst_m, Index dst_n, Index stride_m,
        Index stride_n, Scalar beta, ::nnhaul::Handle & dst)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->src1_m = src1_m;
    args->src1_n = src1_n;
    args->src1_channels = src1_channels;
    args->batch = batch;
    args->src2_m = src2_m;
    args->src2_n = src2_n;
    args->dilation_m = dilation_m;
    args->dilation_n = dilation_n;
    args->dst_channels = dst_channels;
    args->offset_m = offset_m;
    args->offset_n = offset_n;
    args->alpha = alpha;
    args->dst_m = dst_m;
    args->dst_n = dst_n;
    args->stride_m = stride_m;
    args->stride_n = stride_n;
    args->beta = beta;
    enum starpu_data_access_mode dst_mode = STARPU_RW;
    if(beta == 0.0)
    {
        dst_mode = STARPU_W;
    }
    // Submit task: staged by design (the family-3 derivation above).
    ::nntile::haul::stage::staged_insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src1 }, { STARPU_R, &src2 }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Conv2dInplace<std::tuple<nntile::fp64_t>>;
template class Conv2dInplace<std::tuple<nntile::fp32_t>>;
template class Conv2dInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Conv2dInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Conv2dInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Conv2dInplace<std::tuple<nntile::bf16_t>>;

//! Pack of conv2d_inplace operations for different types
conv2d_inplace_pack_t conv2d_inplace;

} // namespace nntile::haul
