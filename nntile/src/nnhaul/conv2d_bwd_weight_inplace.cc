/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/conv2d_bwd_weight_inplace.cc
 * Backward 2D-Convolution of two tensors in WHCN format to get grad of weight
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/conv2d_bwd_weight_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/conv2d_bwd_weight_inplace.hh"
#include "nntile/kernel/nnhaul/chunk_ops.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{
// ---------------------------------------------------------------------------
// Native chunked conv2d_bwd_weight_inplace (no staging).
//
// The stock kernel's index map and accumulation order are preserved
// exactly; every global load/store is addressed in the operand's flat
// element space and mapped through the chunk tables (the device entry
// point is ::nntile::kernel::nnhaul::chunk_conv2d_bwd_weight_inplace). Every weight
// output reduces over the whole src1/src2, so a destination piece's
// touched source span is bounded and the host emitter splits the
// destination range until the by-value tables fit. All-unchunked
// buffers keep the single classic kernel call, bitwise.
// ---------------------------------------------------------------------------

//! Constructor
template<typename T>
Conv2dBwdWeightInplace<std::tuple<T>>::Conv2dBwdWeightInplace():
    codelet(
        "nntile_conv2d_bwd_weight_inplace",
        &Conv2dBwdWeightInplace<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Conv2dBwdWeightInplace<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Conv2dBwdWeightInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply conv2d_bwd_weight_inplace on StarPU buffer on CPU
template <typename T>
void Conv2dBwdWeightInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_src1 = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_src2 = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[2]);
    if(buf_src1->chunk_count() == 1 && buf_src2->chunk_count() == 1
        && buf_dst->chunk_count() == 1)
    {
        // Unchunked fast path: the historical single kernel call, bitwise.
        kernel::conv2d_bwd_weight_inplace::cpu<T>(
            args->src1_m,
            args->src1_n,
            args->src1_channels,
            args->batch,
            args->src2_m,
            args->src2_n,
            args->stride_m,
            args->stride_n,
            args->src2_channels,
            args->offset_m,
            args->offset_n,
            args->alpha,
            buf_src1->get_ptr<T>(),
            buf_src2->get_ptr<T>(),
            args->dst_m,
            args->dst_n,
            args->dilation_m,
            args->dilation_n,
            args->beta,
            buf_dst->get_ptr<T>()
        );
        return;
    }
    // Chunked CPU body (never scheduled by this runtime): the stock
    // arithmetic, elementwise through the chunk views.
    using Y = typename T::repr_t;
    chunkunits::ChunkView<T> const v1(buf_src1, 0);
    chunkunits::ChunkView<T> const v2(buf_src2, 0);
    chunkunits::ChunkView<T> const vd(buf_dst, 0);
    auto at = [](chunkunits::ChunkView<T> const &v, std::size_t f) -> T &
    {
        std::size_t const c = v.chunk_of(f);
        return *v.ptr(c, v.chunk_off(f));
    };
    Index const src1_step =
        args->src1_channels * args->src1_n * args->src1_m;
    Index const src2_step =
        args->src2_channels * args->src2_n * args->src2_m;
    auto const dst_lo = std::max(
        (args->offset_m - args->stride_m*(args->src2_m-1)
            + args->dilation_m-1) / args->dilation_m, Index(0));
    auto const dst_hi = std::min(
        (args->src1_m-1 + args->offset_m + args->dilation_m)
            / args->dilation_m, args->dst_m);
    auto const dst_lo_n = std::max(
        (args->offset_n - args->stride_n*(args->src2_n-1)
            + args->dilation_n-1) / args->dilation_n, Index(0));
    auto const dst_hi_n = std::min(
        (args->src1_n-1 + args->offset_n + args->dilation_n)
            / args->dilation_n, args->dst_n);
    for(Index oc = 0; oc < args->src2_channels; ++oc)
    {
        for(Index ic = 0; ic < args->src1_channels; ++ic)
        {
            for(Index dst_j = 0; dst_j < args->dst_n; ++dst_j)
            {
                for(Index dst_i = 0; dst_i < args->dst_m; ++dst_i)
                {
                    std::size_t const df = static_cast<std::size_t>(
                        ((oc*args->src1_channels+ic)*args->dst_n+dst_j)
                        * args->dst_m + dst_i);
                    if(dst_i >= dst_lo and dst_i < dst_hi
                        and dst_j >= dst_lo_n and dst_j < dst_hi_n)
                    {
                        Y conv{0.0}, c{0}, y, t;
                        Index src1_start_m =
                            args->dilation_m*dst_i - args->offset_m;
                        if(src1_start_m < 0)
                        {
                            Index const neg_rem =
                                (-src1_start_m) % args->stride_m;
                            src1_start_m = neg_rem == 0 ? 0
                                : args->stride_m - neg_rem;
                        }
                        Index const src1_end_m = std::min(
                            args->dilation_m*dst_i - args->offset_m
                                + args->stride_m*(args->src2_m-1) + 1,
                            args->src1_m);
                        Index src1_start_n =
                            args->dilation_n*dst_j - args->offset_n;
                        if(src1_start_n < 0)
                        {
                            Index const neg_rem =
                                (-src1_start_n) % args->stride_n;
                            src1_start_n = neg_rem == 0 ? 0
                                : args->stride_n - neg_rem;
                        }
                        Index const src1_end_n = std::min(
                            args->dilation_n*dst_j - args->offset_n
                                + args->stride_n*(args->src2_n-1) + 1,
                            args->src1_n);
                        for(Index src1_i = src1_start_m;
                            src1_i < src1_end_m;
                            src1_i += args->stride_m)
                        {
                            for(Index src1_j = src1_start_n;
                                src1_j < src1_end_n;
                                src1_j += args->stride_n)
                            {
                                std::size_t const s1f =
                                    static_cast<std::size_t>(src1_i
                                        + (ic*args->src1_n+src1_j)
                                            *args->src1_m);
                                std::size_t const s2f =
                                    static_cast<std::size_t>(
                                        (src1_i+args->offset_m
                                            -args->dilation_m*dst_i)
                                            /args->stride_m
                                        + (oc*args->src2_n
                                            + (src1_j+args->offset_n
                                                -args->dilation_n*dst_j)
                                                /args->stride_n)
                                            *args->src2_m);
                                for(Index b = 0; b < args->batch; ++b)
                                {
                                    Y const a{at(v1, s1f
                                        + src1_step*b)};
                                    Y const w{at(v2, s2f
                                        + src2_step*b)};
                                    y = a*w - c;
                                    t = conv + y;
                                    c = (t-conv) - y;
                                    conv = t;
                                }
                            }
                        }
                        if(args->beta == 0.0)
                        {
                            at(vd, df) = T{args->alpha * conv};
                        }
                        else
                        {
                            Y const old{at(vd, df)};
                            at(vd, df) = T{(args->beta*old - args->alpha*c)
                                + args->alpha*conv};
                        }
                    }
                    else if(args->beta == 0.0)
                    {
                        at(vd, df) = T{Y{0.0}};
                    }
                    else if(args->beta != 1.0)
                    {
                        Y const old{at(vd, df)};
                        at(vd, df) = T{args->beta * old};
                    }
                }
            }
        }
    }
}

#ifdef NNTILE_USE_CUDA
//! Apply conv2d_bwd_weight_inplace on StarPU buffer on CUDA
template<typename T>
void Conv2dBwdWeightInplace<std::tuple<T>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_src1 = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_src2 = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[2]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if(buf_src1->chunk_count() == 1 && buf_src2->chunk_count() == 1
        && buf_dst->chunk_count() == 1)
    {
        // Unchunked fast path: the historical single kernel call, bitwise.
        kernel::conv2d_bwd_weight_inplace::cuda<T>(
            stream,
            args->src1_m,
            args->src1_n,
            args->src1_channels,
            args->batch,
            args->src2_m,
            args->src2_n,
            args->stride_m,
            args->stride_n,
            args->src2_channels,
            args->offset_m,
            args->offset_n,
            args->alpha,
            buf_src1->get_ptr<T>(),
            buf_src2->get_ptr<T>(),
            args->dst_m,
            args->dst_n,
            args->dilation_m,
            args->dilation_n,
            args->beta,
            buf_dst->get_ptr<T>()
        );
        return;
    }
    ::nntile::kernel::nnhaul::chunk_conv2d_bwd_weight_inplace(*buf_src1, *buf_src2,
        *buf_dst, args->src1_m, args->src1_n, args->src1_channels,
        args->batch, args->src2_m, args->src2_n, args->stride_m,
        args->stride_n, args->src2_channels, args->offset_m,
        args->offset_n, args->alpha, args->dst_m, args->dst_n,
        args->dilation_m, args->dilation_n, args->beta,
        chunkunits::nnhaul_elem_type<T>());
}
#endif // NNTILE_USE_CUDA

//! Footprint for conv2d_bwd_weight_inplace tasks
template<typename T>
std::uint64_t Conv2dBwdWeightInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over entire args
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(args, sizeof(*args), hash);
    return hash;
}

//! Submit conv2d_bwd_weight_inplace task
template<typename T>
void Conv2dBwdWeightInplace<std::tuple<T>>::submit(int starpu_worker_hint,
    Index src1_m,
    Index src1_n,
    Index src1_channels,
    Index batch,
    Index src2_m,
    Index src2_n,
    Index stride_m,
    Index stride_n,
    Index src2_channels,
    Index offset_m,
    Index offset_n,
    Scalar alpha,
    ::nnhaul::Handle & src1,
    ::nnhaul::Handle & src2,
    Index dst_m,
    Index dst_n,
    Index dilation_m,
    Index dilation_n,
    Scalar beta,
    ::nnhaul::Handle & dst
)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->src1_m = src1_m;
    args->src1_n = src1_n;
    args->src1_channels = src1_channels;
    args->batch = batch;
    args->src2_m = src2_m;
    args->src2_n = src2_n;
    args->stride_m = stride_m;
    args->stride_n = stride_n;
    args->src2_channels = src2_channels;
    args->offset_m = offset_m;
    args->offset_n = offset_n;
    args->alpha = alpha;
    args->dst_m = dst_m;
    args->dst_n = dst_n;
    args->dilation_m = dilation_m;
    args->dilation_n = dilation_n;
    args->beta = beta;
    enum starpu_data_access_mode dst_mode = STARPU_RW;
    if(beta == 0.0)
    {
        dst_mode = STARPU_W;
    }
    // The body picks the unchunked classic call or the native chunked
    // kernel; no staging either way.
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src1 }, { STARPU_R, &src2 }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Conv2dBwdWeightInplace<std::tuple<nntile::fp64_t>>;
template class Conv2dBwdWeightInplace<std::tuple<nntile::fp32_t>>;
template class Conv2dBwdWeightInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Conv2dBwdWeightInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Conv2dBwdWeightInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Conv2dBwdWeightInplace<std::tuple<nntile::bf16_t>>;

//! Pack of conv2d_bwd_weight_inplace operations for different types
conv2d_bwd_weight_inplace_pack_t conv2d_bwd_weight_inplace;

} // namespace nntile::haul
