/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/subcopy.cc
 * Copy subarray based on contiguous indices
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/subcopy.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/subcopy.hh"
#include "nntile/nnhaul/chunk_units.hh"
#include "nntile/nnhaul/pack_args.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked subcopy driver (chunk_units.hh scheme).
//
// Kernel index map (derived from kernel::subcopy::cpu/cuda, NOT
// assumed): the window enumerates copy_shape row-major with dim 0
// FASTEST and stride[0] == 1 on both sides (a kernel assumption, marked
// in its sources), so the enumeration is a matrix of ncols =
// prod(copy_shape[1:]) COLUMN units of C0 = copy_shape[0] CONTIGUOUS
// elements, the column of index c sitting at src offset
// src_start[0] + sum_{d>=1} (src_start[d] + i_d(c)) * src_stride[d]
// (same form on dst). A column is therefore one 1-D kernel sub-call
// (ndim'=1, copy_shape'={C0}, strides {1}) placed at its two base
// pointers; when both sides' stride[1] equals C0, consecutive columns
// of one dim-1 run are FLAT-adjacent on both sides, and a whole run of
// columns inside single chunks of both buffers is one 2-D sub-call
// (ndim'=2, copy_shape'={C0, L}, strides {1, C0}) - the piece is also
// cut at dim-1 run boundaries so its enumeration carries never leave
// the kernel sub-window's own index arithmetic. A column straddling a
// chunk boundary on either side is cut at that boundary into 1-D
// sub-calls (both sides advance by the same in-column offset o). A
// pure element copy, so the piecewise result is BITWISE the plain call
// under any factorization; strided (non-dense) windows simply degrade
// to one piece per column, which the redistribution use case accepts.
// Zero new kernels; all-unchunked buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Sub>
void subcopy_chunked(Index ndim, const Index *src_start,
    const Index *src_stride, const Index *copy_shape, const Index *dst_start,
    const Index *dst_stride, ::nnhaul::Buffer *buf_src,
    ::nnhaul::Buffer *buf_dst, std::size_t force_bytes, Sub &&sub)
{
    if(ndim <= 0 || copy_shape[0] <= 0)
    {
        return;
    }
    Index const C0 = copy_shape[0];
    Index ncols = 1;
    for(Index d = 1; d < ndim; ++d)
    {
        if(copy_shape[d] <= 0)
        {
            return;
        }
        ncols *= copy_shape[d];
    }
    chunkunits::ChunkView<T> const cv_src(buf_src, force_bytes);
    chunkunits::ChunkView<T> const cv_dst(buf_dst, force_bytes);
    bool const batchable = ndim >= 2 && src_stride[1] == C0
        && dst_stride[1] == C0;
    Index idx[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    Index const mdims = ndim - 1;
    Index sstart1[1] = {0};
    Index sshape1[1] = {1};
    Index sstride1[1] = {1};
    Index sstart2[2] = {0, 0};
    Index sshape2[2] = {1, 1};
    Index sstride2[2] = {1, 1};
    auto col_offsets = [&](Index const *ix, Index &so, Index &do_)
    {
        so = src_start[0];
        do_ = dst_start[0];
        for(Index d = 1; d < ndim; ++d)
        {
            so += (src_start[d] + ix[d - 1]) * src_stride[d];
            do_ += (dst_start[d] + ix[d - 1]) * dst_stride[d];
        }
    };
    Index c = 0;
    while(c < ncols)
    {
        Index so = 0, dof = 0;
        col_offsets(idx, so, dof);
        std::size_t const fs = std::size_t(so);
        std::size_t const fd = std::size_t(dof);
        bool const str_src = !chunkunits::in_one_chunk(cv_src, fs,
            std::size_t(C0));
        bool const str_dst = !chunkunits::in_one_chunk(cv_dst, fd,
            std::size_t(C0));
        if(str_src || str_dst || !batchable)
        {
            // Per-column (or partial-column) 1-D sub-calls, cut at both
            // sides' chunk boundaries.
            Index o = 0;
            while(o < C0)
            {
                std::size_t const as = fs + o;
                std::size_t const ad = fd + o;
                std::size_t const cs = cv_src.chunk_of(as);
                std::size_t const cd = cv_dst.chunk_of(ad);
                Index len = C0 - o;
                Index const cap_s = static_cast<Index>(cv_src.chunk_end(cs)
                    - as);
                Index const cap_d = static_cast<Index>(cv_dst.chunk_end(cd)
                    - ad);
                if(cap_s < len)
                {
                    len = cap_s;
                }
                if(cap_d < len)
                {
                    len = cap_d;
                }
                if(len < 1)
                {
                    len = 1;
                }
                sstart1[0] = 0;
                sshape1[0] = len;
                sub(Index{1}, sstart1, sstride1, sshape1,
                    cv_src.ptr(cs, as - cv_src.chunk_begin(cs)), sstart1,
                    sstride1, cv_dst.ptr(cd, ad - cv_dst.chunk_begin(cd)));
                o += len;
            }
        }
        else
        {
            // Whole-column run inside single chunks of both sides, also
            // confined to the current dim-1 run (no index carry inside).
            std::size_t const cs = cv_src.chunk_of(fs);
            std::size_t const cd = cv_dst.chunk_of(fd);
            Index L = static_cast<Index>(
                (cv_src.chunk_end(cs) - fs < cv_dst.chunk_end(cd) - fd
                    ? cv_src.chunk_end(cs) - fs
                    : cv_dst.chunk_end(cd) - fd) / C0);
            Index const run_left = mdims >= 1 ? copy_shape[1] - idx[0] : 1;
            if(run_left < L)
            {
                L = run_left;
            }
            if(L > ncols - c)
            {
                L = ncols - c;
            }
            if(L < 1)
            {
                L = 1;
            }
            sstart2[0] = 0;
            sstart2[1] = 0;
            sshape2[0] = C0;
            sshape2[1] = L;
            sstride2[0] = 1;
            sstride2[1] = C0;
            sub(Index{2}, sstart2, sstride2, sshape2,
                cv_src.ptr(cs, fs - cv_src.chunk_begin(cs)), sstart2,
                sstride2, cv_dst.ptr(cd, fd - cv_dst.chunk_begin(cd)));
            // Advance L columns through the column-space index.
            idx[0] += L;
            for(Index d = 0; d < mdims - 1; ++d)
            {
                if(idx[d] < copy_shape[d + 1])
                {
                    break;
                }
                idx[d] = 0;
                idx[d + 1] += 1;
            }
            c += L;
            continue;
        }
        // Advance one column through the column-space index.
        ++c;
        if(mdims >= 1)
        {
            idx[0] += 1;
            for(Index d = 0; d < mdims - 1; ++d)
            {
                if(idx[d] < copy_shape[d + 1])
                {
                    break;
                }
                idx[d] = 0;
                idx[d + 1] += 1;
            }
        }
    }
}

} // namespace

//! Constructor
template<typename T>
Subcopy<std::tuple<T>>::Subcopy():
    codelet(
        "nntile_subcopy",
        &Subcopy<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Subcopy<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Subcopy<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Complex copying through StarPU buffers on CPU
template<typename T>
void Subcopy<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    const Index *ndim_ptr, *src_start, *src_stride, *copy_shape, *dst_start,
          *dst_stride;
    ::nntile::haul::unpack_args_ptr(cl_args, ndim_ptr, src_start, src_stride,
            copy_shape, dst_start, dst_stride);
    // Get interfaces
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    if(buf_src->chunk_count() == 1 && buf_dst->chunk_count() == 1
        && forced == 0)
    {
        // Unchunked fast path: the historical single kernel call, bitwise.
        // Kernel maps the scratch into src and dst indexes (2 * ndim).
        std::size_t const scratch_n =
            static_cast<std::size_t>(*ndim_ptr > 0 ? *ndim_ptr : 1);
        std::vector<int64_t> tmp_index_storage(scratch_n * 2);
        int64_t *tmp_index = tmp_index_storage.data();
        // Launch kernel
        kernel::subcopy::cpu<T>(*ndim_ptr, src_start, src_stride,
                copy_shape, buf_src->get_ptr<T>(), dst_start, dst_stride,
                buf_dst->get_ptr<T>(), tmp_index);
        return;
    }
    // Sub-call scratch (2 * ndim' int64, ndim' <= 2).
    std::vector<int64_t> tmp_index_storage(4);
    subcopy_chunked<T>(*ndim_ptr, src_start, src_stride, copy_shape,
        dst_start, dst_stride, buf_src, buf_dst, forced,
        [&](Index sndim, const Index *sstart, const Index *sstride,
            const Index *sshape, T const *psrc, const Index *dstart,
            const Index *dstride, T *pdst) {
            kernel::subcopy::cpu<T>(sndim, sstart, sstride, sshape, psrc,
                dstart, dstride, pdst, tmp_index_storage.data());
        });
}

#ifdef NNTILE_USE_CUDA
//! Complex copying through StarPU buffers on CUDA
template<typename T>
void Subcopy<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    const Index *ndim_ptr, *src_start, *src_stride, *copy_shape, *dst_start,
          *dst_stride;
    ::nntile::haul::unpack_args_ptr(cl_args, ndim_ptr, src_start, src_stride,
            copy_shape, dst_start, dst_stride);
    // Get interfaces
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    if(buf_src->chunk_count() == 1 && buf_dst->chunk_count() == 1
        && forced == 0)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::subcopy::cuda<T>(stream, *ndim_ptr, src_start, src_stride,
            copy_shape, buf_src->get_ptr<T>(), dst_start, dst_stride,
            buf_dst->get_ptr<T>());
        return;
    }
    subcopy_chunked<T>(*ndim_ptr, src_start, src_stride, copy_shape,
        dst_start, dst_stride, buf_src, buf_dst, forced,
        [stream](Index sndim, const Index *sstart, const Index *sstride,
            const Index *sshape, T const *psrc, const Index *dstart,
            const Index *dstride, T *pdst) {
            kernel::subcopy::cuda<T>(stream, sndim, sstart, sstride, sshape,
                psrc, dstart, dstride, pdst);
        });
}
#endif // NNTILE_USE_CUDA

//! Footprint for subcopy tasks that depend on copy shape
template<typename T>
std::uint64_t Subcopy<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    const Index *ndim_ptr, *src_start, *src_stride, *copy_shape, *dst_start,
          *dst_stride;
    ::nntile::haul::unpack_args_ptr(cl_args, ndim_ptr, src_start, src_stride,
            copy_shape, dst_start, dst_stride);
    std::size_t copy_shape_size = *ndim_ptr * sizeof(*copy_shape);
    // Apply hash over parameter copy_shape
    return ::nntile::haul::fnv1a(copy_shape, copy_shape_size, 0);
}

template<typename T>
void Subcopy<std::tuple<T>>::submit(int starpu_worker_hint,
        Index ndim, const std::vector<Index> &src_start,
        const std::vector<Index> &src_stride,
        const std::vector<Index> &dst_start,
        const std::vector<Index> &dst_stride,
        const std::vector<Index> &copy_shape, ::nnhaul::Handle & src, ::nnhaul::Handle & dst,
        ::nnhaul::Handle & tmp_index, starpu_data_access_mode mode)
{
    constexpr double nflops = 0;
    (void)nflops;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked column driver (any window geometry, strided windows simply
    // degrade to per-column pieces); subcopy never stages.
    ::nntile::haul::insert_values(codelet.raw, starpu_worker_hint,
        { { &(ndim), sizeof(ndim) }, { &(src_start[0]), ndim*sizeof(src_start[0]) }, { &(src_stride[0]), ndim*sizeof(src_stride[0]) }, { &(copy_shape[0]), ndim*sizeof(copy_shape[0]) }, { &(dst_start[0]), ndim*sizeof(dst_start[0]) }, { &(dst_stride[0]), ndim*sizeof(dst_stride[0]) } },
        { { STARPU_R, &src }, { mode, &dst } });

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Subcopy<std::tuple<nntile::int64_t>>;
template class Subcopy<std::tuple<nntile::bool_t>>;
template class Subcopy<std::tuple<nntile::fp64_t>>;
template class Subcopy<std::tuple<nntile::fp32_t>>;
template class Subcopy<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Subcopy<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Subcopy<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Subcopy<std::tuple<nntile::bf16_t>>;
template class Subcopy<std::tuple<nntile::fp16_t>>;

//! Pack of subcopy operations for different types
subcopy_pack_t subcopy;

} // namespace nntile::haul
