/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/conv2d_inplace.cc
 * Forward 2D-Convolution of two tensors in WHCN format
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.1.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
// Corresponding header
#include "nntile/nnhaul/ops/conv2d_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/conv2d_inplace.hh"
#include "nntile/nnhaul/chunk_units.hh"
#include <nntile/kernel_scope.hh>

namespace nntile::nnhaul
{
// ---------------------------------------------------------------------------
// Native chunked driver (no staging): the kernel layer's chunked entry
// takes device chunk-pointer tables for src1, src2 and dst and addresses
// every operand flat offset with the power-of-two class shift/mask, so the
// whole destination range is ONE launch. The per-element arithmetic is the
// classic kernel's (same bounds, same loop order, same Kahan sum), hence
// bitwise. All-unchunked buffers keep the classic whole-buffer call.
//
// (Historical staged derivation, retained for context: conv2d_inplace was
// not cleanly decomposable with the whole-buffer kernel signature.
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
// OVERLAP, so staged windows cannot retire independently). Transformer workloads do
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

//! Apply conv2d_inplace on NNHaul buffer on CPU
template<typename T>
void Conv2dInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    const T *src1 = ::nntile::nnhaul::buf_as<T>(buffers, 0);
    const T *src2 = ::nntile::nnhaul::buf_as<T>(buffers, 1);
    T *dst = ::nntile::nnhaul::buf_as<T>(buffers, 2);
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
//! Apply conv2d_inplace on NNHaul buffer on CUDA
template<typename T>
void Conv2dInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // The kernel derives its flat index and grid from this output count in
    // 32-bit int.
    chunkunits::require_kernel_index_envelope(
        static_cast<std::size_t>(args->dst_m) * args->dst_n
            * args->dst_channels * args->batch + 255,
        "nntile_conv2d_inplace");
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
            buf_src1->get_ptr<T>(),
            buf_src2->get_ptr<T>(),
            args->dst_m,
            args->dst_n,
            args->stride_m,
            args->stride_n,
            args->beta,
            buf_dst->get_ptr<T>());
        return;
    }
    // Chunked single-launch path.
    chunkunits::ChunkView<T> const cv_src1(buf_src1);
    chunkunits::ChunkView<T> const cv_src2(buf_src2);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    chunkunits::LaunchTables &tabs = chunkunits::launch_tables();
    tabs.clear();
    std::size_t const o_src1 = tabs.add(cv_src1);
    std::size_t const o_src2 = tabs.add(cv_src2);
    std::size_t const o_dst = tabs.add(cv_dst);
    chunkunits::ChunkPtrTable &table = chunkunits::chunk_ptr_table();
    if(table.begin(stream))
    {
        void const *table_base = table.upload_tables(tabs);
        if(table_base != nullptr)
        {
            kernel::conv2d_inplace::chunked<T>(
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
                table.table_view<T const *>(table_base, o_src1),
                Index(cv_src1.nchunks),
                table.table_view<T const *>(table_base, o_src2),
                Index(cv_src2.nchunks),
                table.table_view<T *>(table_base, o_dst),
                Index(cv_dst.nchunks),
                args->dst_m,
                args->dst_n,
                args->stride_m,
                args->stride_n,
                args->beta,
                Index(cv_src1.chunk_elems),
                Index(cv_src2.chunk_elems),
                Index(cv_dst.chunk_elems));
            table.end(stream);
            return;
        }
    }
    std::fprintf(stderr,
        "nntile_conv2d_inplace: chunk-ptr table cannot serve this launch "
        "(invariant violation; no staging fallback exists)\n");
    std::abort();
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
    hash = ::nntile::nnhaul::fnv1a(args, sizeof(*args), hash);
    return hash;
}

//! Submit conv2d_inplace task
template<typename T>
void Conv2dInplace<std::tuple<T>>::submit(int worker_hint, Index src1_m, Index src1_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index dilation_m, Index dilation_n,
        Index dst_channels, Index offset_m, Index offset_n, Scalar alpha,
        ::nnhaul::Handle & src1, ::nnhaul::Handle & src2, Index dst_m, Index dst_n, Index stride_m,
        Index stride_n, Scalar beta, ::nnhaul::Handle & dst)
{
    ::nntile::require_kernel_family("conv2d_inplace");
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
    enum data_access_mode dst_mode = ACCESS_RW;
    if(beta == 0.0)
    {
        dst_mode = ACCESS_W;
    }
    // Submit task: the CUDA body serves chunked and unchunked operands
    // natively (no staging).
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_R, &src1 }, { ACCESS_R, &src2 }, { dst_mode, &dst } }, args, sizeof(*args));
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

} // namespace nntile::nnhaul
