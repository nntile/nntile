/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/mask_scalar.cc
 * StarPU wrappers for mask_scalar operation
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/mask_scalar.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/mask_scalar.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked mask_scalar driver (chunk_units.hh scheme).
//
// Kernel index map (derived from kernel::mask_scalar::cpu/cuda, NOT
// assumed): data is nrows-by-ncols Fortran with element (i, j) at flat
// i + j*nrows - the CONTIGUOUS unit is the nrows-element COLUMN, and the
// mask is the small operand indexed by the row only (constant along the
// contiguous axis), exactly the slice-class geometry. A run of whole
// columns [j0, j0+L) inside one data chunk is therefore the plain kernel
// sub-call (nrows, L, mask, val, data + j0*nrows). A column straddling a
// data chunk boundary is staged through a bounded nrows temp (gathered -
// the op writes only masked-out rows - kernel (nrows, 1) on the temp,
// scattered back). The mask itself is one nrows-element segment: used in
// place when its range lies in one chunk, gathered ONCE into a bounded
// temp otherwise (every sub-call reads the whole mask). Writes are
// idempotent val assignments, so the piecewise result is BITWISE the
// plain call under any factorization. Zero new kernels; all-unchunked
// buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void mask_scalar_chunked(Index nrows, Index ncols,
    ::nnhaul::Buffer *buf_mask, Scalar val, ::nnhaul::Buffer *buf_data,
    std::size_t force_bytes, Arena &arena, Sub &&sub)
{
    if(nrows <= 0 || ncols <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_data(buf_data, force_bytes);
    chunkunits::ChunkView<nntile::bool_t> const cv_mask(buf_mask,
        force_bytes);
    bool const chunked_data = cv_data.nchunks > 1 || force_bytes != 0;
    T *temp_col = chunked_data ? arena.alloc(std::size_t(nrows)) : nullptr;
    nntile::bool_t const *mask = nullptr;
    nntile::bool_t *temp_mask = nullptr;
    if(cv_mask.nchunks > 1
        && !chunkunits::in_one_chunk(cv_mask, 0, std::size_t(nrows)))
    {
        // The arena hands out T-sized blocks; the mask needs nrows bytes,
        // so round up in T units and view the block as bool_t (1-byte
        // alignment inside a T alignment).
        std::size_t const t_elems = (std::size_t(nrows) * sizeof(nntile::bool_t)
            + sizeof(T) - 1) / sizeof(T);
        temp_mask = reinterpret_cast<nntile::bool_t *>(arena.alloc(t_elems));
        chunkunits::gather_range(cv_mask, 0, std::size_t(nrows), temp_mask,
            arena);
        mask = temp_mask;
    }
    else
    {
        mask = cv_mask.ptr(0, 0);
    }
    if(!arena.ok_state() || mask == nullptr)
    {
        return;
    }
    Index j = 0;
    while(j < ncols && arena.ok_state())
    {
        std::size_t const f = std::size_t(j) * nrows;
        if(!chunkunits::in_one_chunk(cv_data, f, std::size_t(nrows)))
        {
            // Straddling column: stage it whole through the bounded temp.
            chunkunits::gather_range(cv_data, f, std::size_t(nrows),
                temp_col, arena);
            if(arena.ok_state())
            {
                sub(nrows, Index{1}, mask, val, temp_col);
                chunkunits::scatter_range(cv_data, f, std::size_t(nrows),
                    temp_col, arena);
            }
            ++j;
            continue;
        }
        std::size_t const c = cv_data.chunk_of(f);
        std::size_t const end = cv_data.chunk_end(c);
        Index L = static_cast<Index>((end - f) / nrows);
        if(L > ncols - j)
        {
            L = ncols - j;
        }
        if(L < 1)
        {
            L = 1;
        }
        sub(nrows, L, mask, val, cv_data.ptr(c, f - cv_data.chunk_begin(c)));
        j += L;
    }
}

} // namespace

//! Constructor
template<typename T>
MaskScalar<std::tuple<T>>::MaskScalar():
    codelet(
        "nntile_mask_scalar",
        &MaskScalar<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &MaskScalar<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &MaskScalar<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Mask scalar operation for StarPU buffer on CPU
template<typename T>
void MaskScalar<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_data = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_mask = static_cast<::nnhaul::Buffer *>(buffers[1]);
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    if(buf_data->chunk_count() == 1 && buf_mask->chunk_count() == 1
        && forced == 0)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::mask_scalar::cpu<T>(args->nrows, args->ncols,
            buf_mask->get_ptr<nntile::bool_t>(), args->val,
            buf_data->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        mask_scalar_chunked<T>(
            args->nrows,
            args->ncols,
            buf_mask,
            args->val,
            buf_data,
            forced,
            arena,
            [](Index r, Index c, nntile::bool_t const *pm, Scalar v, T *pd) {
                kernel::mask_scalar::cpu<T>(r, c, pm, v, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result partially unwritten (same failure class as a
        // failed kernel launch).
        std::fprintf(stderr,
            "nntile_mask_scalar: cpu chunked driver aborted after an "
            "allocation failure; data left unwritten\n");
    }
}
// Specializations of CPU wrapper for accelerated types
template<>
void MaskScalar<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaskScalar<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MaskScalar<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaskScalar<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MaskScalar<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaskScalar<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Mask scalar StarPU buffer on CUDA
template<typename T>
void MaskScalar<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_data = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_mask = static_cast<::nnhaul::Buffer *>(buffers[1]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    if(buf_data->chunk_count() == 1 && buf_mask->chunk_count() == 1
        && forced == 0)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::mask_scalar::cuda<T>(stream, args->nrows, args->ncols,
            buf_mask->get_ptr<nntile::bool_t>(), args->val,
            buf_data->get_ptr<T>());
        return;
    }
    chunkunits::ArenaCuda<T> arena(stream);
    mask_scalar_chunked<T>(
        args->nrows,
        args->ncols,
        buf_mask,
        args->val,
        buf_data,
        forced,
        arena,
        [stream](Index r, Index c, nntile::bool_t const *pm, Scalar v,
            T *pd) {
            kernel::mask_scalar::cuda<T>(stream, r, c, pm, v, pd);
        });
    if(!arena.ok_state())
    {
        std::fprintf(stderr,
            "nntile_mask_scalar: chunked CUDA driver aborted after a stream "
            "error; data left unwritten\n");
    }
}
// Specializations of CUDA wrapper for accelerated types
template<>
void MaskScalar<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaskScalar<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void MaskScalar<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaskScalar<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void MaskScalar<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaskScalar<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for mask_scalar tasks
template<typename T>
std::uint64_t MaskScalar<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nrows, sizeof(args->nrows), hash);
    hash = ::nntile::haul::fnv1a(&args->ncols, sizeof(args->ncols), hash);
    return hash;
}

template<typename T>
void MaskScalar<std::tuple<T>>::submit(int starpu_worker_hint,
        Index nrows, Index ncols, ::nnhaul::Handle & mask, Scalar val, ::nnhaul::Handle & data)
//! Insert mask_scalar task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nrows = nrows;
    args->ncols = ncols;
    args->val = val;
    // Indicate maximal possible amount of writes as flops count
    double nflops = sizeof(T) * nrows * (ncols+1);
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked column driver; mask_scalar never stages.
    (void)nflops;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_RW, &data }, { STARPU_R, &mask } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class MaskScalar<std::tuple<nntile::fp64_t>>;
template class MaskScalar<std::tuple<nntile::fp32_t>>;
template class MaskScalar<std::tuple<nntile::fp32_fast_tf32_t>>;
template class MaskScalar<std::tuple<nntile::fp32_fast_fp16_t>>;
template class MaskScalar<std::tuple<nntile::fp32_fast_bf16_t>>;
template class MaskScalar<std::tuple<nntile::bf16_t>>;
template class MaskScalar<std::tuple<nntile::fp16_t>>;

//! Pack of mask_scalar operations for different types
mask_scalar_pack_t mask_scalar;

} // namespace nntile::haul
