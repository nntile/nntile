/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/randn.cc
 * Randn operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/randn.hh"

// Standard libraries
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/randn.hh"
#include "nntile/kernel/nnhaul/chunk_ops.hh"
#include "nntile/nnhaul/chunk_units.hh"
#include "nntile/nnhaul/pack_args.hh"
#include "random.h"


namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked randn driver (chunk_units.hh scheme) for every output
// stride: the dense Fortran geometry uses the column sub-calls below,
// a foreign strided geometry uses the kernel's own strided walk (CUDA:
// ::nntile::kernel::nnhaul::chunk_randn, one position-seeded value per flat position;
// CPU: the classic jump-chained loop with the write offset mapped
// through the chunk view). Nothing is staged.
//
// Generator reproducibility (derived from the kernels, NOT assumed): the
// chameleon PRNG value of an output element depends ONLY on (seed, the
// element's absolute linear position in the UNDERLYING array). The CUDA
// kernel states it structurally - every thread computes
// under = sum_d (coord_d + start_d) * prod(underlying_shape[0:d]) and
// draws from jump(under, seed) - and the CPU kernel realizes the same
// stream by jump-chaining: its initial jump lands on the underlying
// position of start, one randn value advances the doubled LCG by one
// jump unit, and every inter-column jump is the underlying-position
// gap (the kernel's own contract: "randomly generate any array in
// parallel by parts"). A sub-call therefore reproduces the SAME
// per-position value stream whenever it addresses a window of the
// underlying array whose covered positions are CONSECUTIVE along the
// column enumeration.
//
// Kernel index map: with dense Fortran strides (stride[0] == 1,
// stride[d] == stride[d-1]*shape[d-1] - the Tile payload geometry,
// checked at submit), the (nrows=shape[0])-element columns of the
// kernel's matrix view are CONTIGUOUS flat runs, column c covering flat
// [c*nrows, (c+1)*nrows). Consecutive columns' underlying positions are
// consecutive only within one dim-1 run (i1 does not wrap), so:
//  - whole-column pieces are 2-D sub-calls (m=nrows, L columns)
//    reading start' = (start[0], col_lin(c0)) with underlying_shape' =
//    (underlying_shape[0], anything - the kernel never reads dim 1 of a
//    2-D underlying shape for the jump), shape' = (nrows, L),
//    stride' = (1, nrows), placed at the piece's chunk pointer;
//  - a column straddling a chunk boundary is cut at the boundary into
//    1-D sub-calls (start' = absolute underlying position of its first
//    row, shape' = (len), stride' = (1)) - the LCG advance per value
//    makes the k-th draw of the sub-call the (abs_pos+k)-th draw of
//    the stream, exactly the original element's value.
// Writes are position-seeded and independent, so the piecewise result
// is BITWISE the plain call under any factorization. An all-unchunked
// buffer keeps the single classic call, bitwise.
// ---------------------------------------------------------------------------

namespace
{

//! True when the strides are the dense Fortran strides of shape (the
//! flat buffer order then equals the kernel's enumeration order). The
//! ndim <= 8 bound mirrors the CUDA kernel's own MAX_NDIM.
bool dense_fortran_strides(Index ndim, const Index *shape,
    const Index *stride)
{
    if(ndim <= 0 || ndim > 8 || stride[0] != 1)
    {
        return false;
    }
    Index acc = 1;
    for(Index d = 1; d < ndim; ++d)
    {
        acc *= shape[d - 1];
        if(stride[d] != acc)
        {
            return false;
        }
    }
    return true;
}

//! One chameleon normal draw from the current LCG state (the exact CPU
//! kernel arithmetic: CORE_slaran for float repr, CORE_dlaran for
//! double repr).
template<typename Y>
Y randn_value(unsigned long long &seed, Y mean, Y stddev)
{
    constexpr Y two = 2.0;
    constexpr Y twopi = 6.2831853071795864769252867663;
    if constexpr(std::is_same_v<Y, double>)
    {
        Y const t1 = CORE_dlaran(&seed);
        Y const t2 = CORE_dlaran(&seed) * twopi;
        Y const t3 = std::sqrt(-two*std::log(t1)) * std::cos(t2);
        return stddev*t3 + mean;
    }
    else
    {
        Y const t1 = CORE_slaran(&seed);
        Y const t2 = CORE_slaran(&seed) * twopi;
        Y const t3 = std::sqrt(-two*std::log(t1)) * std::cos(t2);
        return stddev*t3 + mean;
    }
}

/*! The classic CPU randn walk with the output offset mapped through the
 * chunk view. The jump-chained LCG state, the column enumeration and
 * every stride advance are the stock kernel's, so the result is
 * BITWISE the classic call; only the destination address changes. */
template<typename T>
void randn_strided_chunked(Index ndim, Index nelems,
    unsigned long long seed, Scalar mean_, Scalar stddev_,
    const Index *start, const Index *shape, const Index *underlying_shape,
    const Index *stride, ::nnhaul::Buffer *buf_data)
{
    using Y = typename T::repr_t;
    Y const mean{mean_};
    Y const stddev{stddev_};
    if(ndim <= 0 || nelems <= 0 || shape[0] <= 0)
    {
        return;
    }
    Index shift = start[ndim-1];
    for(Index i = ndim-2; i >= 0; --i)
    {
        shift = start[i] + shift*underlying_shape[i];
    }
    seed = CORE_rnd64_jump(shift, seed);
    chunkunits::ChunkView<T> const cv(buf_data, 0);
    std::ptrdiff_t f = 0;
    auto put = [&](Y v)
    {
        std::size_t const at = static_cast<std::size_t>(f);
        std::size_t const c = cv.chunk_of(at);
        *cv.ptr(c, cv.chunk_off(at)) = T{v};
    };
    Index const nrows = shape[0];
    Index const ncols = nelems / nrows;
    for(Index i = 0; i < nrows; ++i)
    {
        put(randn_value(seed, mean, stddev));
        f += stride[0];
    }
    std::vector<Index> tmp_index(static_cast<std::size_t>(2*ndim), 0);
    for(Index j = 1; j < ncols; ++j)
    {
        ++tmp_index[1];
        Index k = 1;
        Index col_shift = underlying_shape[0] - nrows;
        Index underlying_stride = underlying_shape[0];
        f += stride[1] - stride[0]*nrows;
        while(tmp_index[k] == shape[k])
        {
            tmp_index[k] = 0;
            ++k;
            ++tmp_index[k];
            col_shift += underlying_stride
                * (underlying_shape[k-1]-shape[k-1]);
            underlying_stride *= underlying_shape[k-1];
            f += stride[k] - stride[k-1]*shape[k-1];
        }
        seed = CORE_rnd64_jump(col_shift, seed);
        for(Index i = 0; i < nrows; ++i)
        {
            put(randn_value(seed, mean, stddev));
            f += stride[0];
        }
    }
}

template<typename T, typename Sub>
void randn_chunked(Index ndim, Index nelems, unsigned long long seed,
    Scalar mean, Scalar stddev, const Index *start, const Index *shape,
    const Index *underlying_shape, ::nnhaul::Buffer *buf_data,
    std::size_t force_bytes, Sub &&sub)
{
    if(ndim <= 0 || nelems <= 0 || shape[0] <= 0)
    {
        return;
    }
    Index const nrows = shape[0];
    Index const ncols = nelems / nrows;
    if(ncols <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv(buf_data, force_bytes);
    Index const U0 = underlying_shape[0];
    // Column-space index bookkeeping: idx[] is the multi-index of the
    // current column over shape[1..ndim-1], us[d] the underlying stride
    // of column-space dim d (us[1] = 1, us[d] = prod(underlying_shape
    // [1..d-1])), col_lin the underlying linear column index.
    Index idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    Index us[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    Index const mdims = ndim - 1 < 7 ? ndim - 1 : 7;
    for(Index d = 2; d <= mdims; ++d)
    {
        Index acc = 1;
        for(Index e = 1; e < d; ++e)
        {
            acc *= underlying_shape[e];
        }
        us[d] = acc;
    }
    Index col_lin = 0;
    for(Index d = 1; d <= mdims; ++d)
    {
        col_lin += (start[d] + idx[d]) * us[d];
    }
    Index start2[2] = {start[0], col_lin};
    Index shape2[2] = {nrows, 1};
    Index under2[2] = {U0, col_lin + 1};
    Index stride2[2] = {1, nrows};
    Index start1[1] = {0};
    Index shape1[1] = {1};
    Index under1[1] = {1};
    Index stride1[1] = {1};
    Index c = 0;
    while(c < ncols)
    {
        std::size_t const flat = std::size_t(c) * nrows;
        if(!chunkunits::in_one_chunk(cv, flat, std::size_t(nrows)))
        {
            // Straddling column: 1-D sub-calls over its row-ranges,
            // seeded by each range's absolute underlying position.
            Index o = 0;
            while(o < nrows)
            {
                std::size_t const at = flat + o;
                std::size_t const cc = cv.chunk_of(at);
                Index const len = static_cast<Index>(
                    cv.chunk_end(cc) - at < std::size_t(nrows) - o
                        ? cv.chunk_end(cc) - at
                        : std::size_t(nrows) - o);
                start1[0] = (start[0] + o) + U0 * col_lin;
                shape1[0] = len;
                under1[0] = start1[0] + len;
                sub(Index{1}, len, seed, mean, stddev, start1, shape1,
                    under1, cv.ptr(cc, at - cv.chunk_begin(cc)), stride1);
                o += len;
            }
            ++c;
            // Advance one column (the straddling column was whole).
            idx[1] += 1;
            col_lin += 1;
            for(Index d = 1; d < mdims; ++d)
            {
                if(idx[d] < shape[d])
                {
                    break;
                }
                idx[d] = 0;
                idx[d + 1] += 1;
            }
            if(mdims >= 1)
            {
                col_lin = 0;
                for(Index d = 1; d <= mdims; ++d)
                {
                    col_lin += (start[d] + idx[d]) * us[d];
                }
            }
            continue;
        }
        std::size_t const cc = cv.chunk_of(flat);
        Index L = static_cast<Index>((cv.chunk_end(cc) - flat) / nrows);
        // Confine to the current dim-1 run (consecutive underlying
        // column positions) and to the remaining columns.
        Index const run_left = mdims >= 1 ? shape[1] - idx[1] : 1;
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
        start2[0] = start[0];
        start2[1] = col_lin;
        shape2[0] = nrows;
        shape2[1] = L;
        under2[0] = U0;
        under2[1] = col_lin + L;
        sub(Index{2}, nrows * L, seed, mean, stddev, start2, shape2, under2,
            cv.ptr(cc, flat - cv.chunk_begin(cc)), stride2);
        c += L;
        // Advance the column-space index by L columns with carries.
        if(mdims >= 1)
        {
            idx[1] += L;
            for(Index d = 1; d < mdims; ++d)
            {
                if(idx[d] < shape[d])
                {
                    break;
                }
                idx[d] = 0;
                idx[d + 1] += 1;
            }
            if(idx[mdims] >= shape[mdims] && ncols - c > 0)
            {
                // Should not happen: L was confined to the total column
                // count; guard anyway.
                idx[mdims] = 0;
            }
            col_lin = 0;
            for(Index d = 1; d <= mdims; ++d)
            {
                col_lin += (start[d] + idx[d]) * us[d];
            }
        }
    }
}

} // namespace

//! Constructor
template<typename T>
Randn<std::tuple<T>>::Randn():
    codelet(
        "nntile_randn",
        &Randn<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Randn<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Randn<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::randn::cpu<T>
template<typename T>
void Randn<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    const Index *ndim_ptr, *nelems_ptr, *start, *shape, *stride,
          *underlying_shape;
    const unsigned long long *seed_ptr;
    const Scalar *mean_ptr, *stddev_ptr;
    ::nntile::haul::unpack_args_ptr(cl_args, ndim_ptr, nelems_ptr, seed_ptr, mean_ptr,
            stddev_ptr, start, shape, stride, underlying_shape);
    // Get interfaces
    Index ndim = *ndim_ptr;
    auto buf_data = static_cast<::nnhaul::Buffer *>(buffers[0]);
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    bool const dense = dense_fortran_strides(ndim, shape, stride);
    if(buf_data->chunk_count() == 1 && (forced == 0 || !dense))
    {
        // Unchunked fast path: the historical single kernel call, bitwise.
        // (A forced virtual factorization of a non-dense geometry cannot
        // use the native driver and keeps this call.)
        // Index walk writes one past ndim; StarPU scratch is 2 * ndim.
        std::size_t const scratch_n =
            static_cast<std::size_t>(*ndim_ptr > 0 ? *ndim_ptr : 1);
        std::vector<nntile::int64_t> tmp_index_storage(scratch_n * 2);
        nntile::int64_t *tmp_index = tmp_index_storage.data();
        // Launch kernel
        kernel::randn::cpu<T>(
            ndim,
            *nelems_ptr,
            *seed_ptr,
            *mean_ptr,
            *stddev_ptr,
            start,
            shape,
            underlying_shape,
            buf_data->get_ptr<T>(),
            stride,
            tmp_index
        );
        return;
    }
    if(dense)
    {
        // Sub-call scratch (2 * ndim' int64, ndim' <= 2).
        std::vector<nntile::int64_t> tmp_index_storage(4);
        randn_chunked<T>(
            ndim,
            *nelems_ptr,
            *seed_ptr,
            *mean_ptr,
            *stddev_ptr,
            start,
            shape,
            underlying_shape,
            buf_data,
            forced,
            [&](Index sndim, Index snelems, unsigned long long sseed,
                Scalar smean, Scalar sstddev, const Index *sstart,
                const Index *sshape, const Index *sunder, T *sdata,
                const Index *sstride) {
                kernel::randn::cpu<T>(
                    sndim,
                    snelems,
                    sseed,
                    smean,
                    sstddev,
                    sstart,
                    sshape,
                    sunder,
                    sdata,
                    sstride,
                    tmp_index_storage.data());
            });
    }
    else
    {
        // Foreign strided output on a chunked buffer: the classic
        // jump-chained walk with the destination offset mapped through
        // the chunk view (bitwise the classic call).
        randn_strided_chunked<T>(
            ndim,
            *nelems_ptr,
            *seed_ptr,
            *mean_ptr,
            *stddev_ptr,
            start,
            shape,
            underlying_shape,
            stride,
            buf_data);
    }
}

#ifdef NNTILE_USE_CUDA
//! NNHaul wrapper for kernel::randn::cuda<T>
template<typename T>
void Randn<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    const Index *ndim_ptr, *nelems_ptr, *start, *shape, *stride,
          *underlying_shape;
    const unsigned long long *seed_ptr;
    const Scalar *mean_ptr, *stddev_ptr;
    ::nntile::haul::unpack_args_ptr(
        cl_args,
        ndim_ptr,
        nelems_ptr,
        seed_ptr,
        mean_ptr,
        stddev_ptr,
        start,
        shape,
        stride,
        underlying_shape
    );
    Index ndim = *ndim_ptr;
    auto buf_data = static_cast<::nnhaul::Buffer *>(buffers[0]);
    cudaStream_t stream = ::nnhaul::cuda_stream();
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    bool const dense = dense_fortran_strides(ndim, shape, stride);
    if(buf_data->chunk_count() == 1 && (forced == 0 || !dense))
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::randn::cuda<T>(
            stream,
            ndim,
            *nelems_ptr,
            *seed_ptr,
            *mean_ptr,
            *stddev_ptr,
            start,
            shape,
            underlying_shape,
            buf_data->get_ptr<T>(),
            stride
        );
        return;
    }
    if(dense)
    {
        randn_chunked<T>(
            ndim,
            *nelems_ptr,
            *seed_ptr,
            *mean_ptr,
            *stddev_ptr,
            start,
            shape,
            underlying_shape,
            buf_data,
            forced,
            [stream](Index sndim, Index snelems, unsigned long long sseed,
                Scalar smean, Scalar sstddev, const Index *sstart,
                const Index *sshape, const Index *sunder, T *sdata,
                const Index *sstride) {
                kernel::randn::cuda<T>(
                    stream,
                    sndim,
                    snelems,
                    sseed,
                    smean,
                    sstddev,
                    sstart,
                    sshape,
                    sunder,
                    sdata,
                    sstride);
            });
    }
    else
    {
        // Foreign strided output on a chunked buffer: the stock CUDA
        // kernel's per-position map, with the strided offset through
        // the destination chunk table (bitwise the classic call).
        long long lstart[8];
        long long lshape[8];
        long long lstride[8];
        long long lunder[8];
        Index const n = ndim < 8 ? ndim : 8;
        for(Index d = 0; d < n; ++d)
        {
            lstart[d] = start[d];
            lshape[d] = shape[d];
            lstride[d] = stride[d];
            lunder[d] = underlying_shape[d];
        }
        ::nntile::kernel::nnhaul::chunk_randn(*buf_data,
            static_cast<std::size_t>(*nelems_ptr), *seed_ptr, *mean_ptr,
            *stddev_ptr, n, lstart, lshape, lstride, lunder,
            chunkunits::nnhaul_elem_type<T>());
    }
}

template<>
void Randn<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    Randn<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Randn<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    Randn<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Randn<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    Randn<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for randn tasks that depend on shape
template<typename T>
std::uint64_t Randn<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    const Index *ndim_ptr, *nelems_ptr, *start, *shape, *stride,
          *underlying_shape;
    const unsigned long long *seed_ptr;
    const Scalar *mean_ptr, *stddev_ptr;
    ::nntile::haul::unpack_args_ptr(cl_args, ndim_ptr, nelems_ptr, seed_ptr,
            mean_ptr, stddev_ptr, start, shape, stride, underlying_shape);
    std::size_t shape_size = *ndim_ptr * sizeof(*shape);
    // Apply hash over parameter copy_shape
    return ::nntile::haul::fnv1a(shape, shape_size, 0);
}

template<typename T>
void Randn<std::tuple<T>>::submit(int starpu_worker_hint,
    Index ndim,
    Index nelems,
    unsigned long long seed,
    Scalar mean,
    Scalar stddev,
    const std::vector<Index> &start,
    const std::vector<Index> &shape,
    const std::vector<Index> &stride,
    const std::vector<Index> &underlying_shape,
    ::nnhaul::Handle & data
)
{
    double nflops = 2 * nelems;
    (void)nflops;
    // The body picks the unchunked classic call or the native chunked
    // driver for BOTH dense and foreign strided geometries; no staging.
    ::nntile::haul::insert_values(codelet.raw, starpu_worker_hint,
        { { &ndim, sizeof(ndim) }, { &nelems, sizeof(nelems) }, { &seed, sizeof(seed) }, { &mean, sizeof(mean) }, { &stddev, sizeof(stddev) }, { &start[0], ndim*sizeof(start[0]) }, { &shape[0], ndim*sizeof(shape[0]) }, { &stride[0], ndim*sizeof(stride[0]) }, { &underlying_shape[0], ndim*sizeof(underlying_shape[0]) } },
        { { STARPU_W, &data } });
}
// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Randn<std::tuple<nntile::fp64_t>>;
template class Randn<std::tuple<nntile::fp32_t>>;
template class Randn<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Randn<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Randn<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Randn<std::tuple<nntile::bf16_t>>;

//! Pack of randn operations for different types
randn_pack_t randn;

} // namespace nntile::haul
