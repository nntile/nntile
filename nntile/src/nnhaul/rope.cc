/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/rope.cc
 * Rotary positional embedding
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/rope.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/rope.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked rope driver (chunk_units.hh scheme).
//
// Kernel index map (derived from kernel::rope::cpu/cuda, NOT assumed):
// src/dst are the 2-by-m-by-n pair tensor with the pair of position
// (i, j) at flat 2*(i + j*m) - so the big tile is n COLUMN units of 2*m
// CONTIGUOUS elements (the pair of in-column position i sits at the
// in-column offset 2*i), and sin/cos are the small per-position
// operands (m elements each, constant along the column axis - the
// slice-class geometry; the pair stride is 2 WITHIN a column, so the
// contiguous-run unit is the whole column). A run of whole columns
// [j0, j0+L) inside single chunks of BOTH src and dst is therefore the
// plain kernel sub-call (m, L, sin, cos, src + 2*m*j0, dst + 2*m*j0).
// A column straddling a src or dst chunk boundary is staged through a
// bounded 2*m temp pair (src gathered, dst gathered-and-scattered since
// the kernel writes every pair); the sin/cos tables are used in place,
// or gathered ONCE into bounded m temps when their own ranges
// straddle. Pure elementwise pairwise rotation, so the piecewise
// result is BITWISE the plain call under any factorization. Zero new
// kernels; all-unchunked buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void rope_chunked(Index m, Index n, ::nnhaul::Buffer *buf_sin,
    ::nnhaul::Buffer *buf_cos, ::nnhaul::Buffer *buf_src,
    ::nnhaul::Buffer *buf_dst, Arena &arena,
    Sub &&sub)
{
    if(m <= 0 || n <= 0)
    {
        return;
    }
    std::size_t const col = 2 * std::size_t(m);
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    chunkunits::ChunkView<T> const cv_sin(buf_sin);
    chunkunits::ChunkView<T> const cv_cos(buf_cos);
    bool const chunked_big =
        cv_src.nchunks > 1 || cv_dst.nchunks > 1;
    T *temp_src = chunked_big ? arena.alloc(col) : nullptr;
    T *temp_dst = chunked_big ? arena.alloc(col) : nullptr;
    T const *psin = nullptr;
    T const *pcos = nullptr;
    if(cv_sin.nchunks > 1
        && !chunkunits::in_one_chunk(cv_sin, 0, std::size_t(m)))
    {
        T *t = arena.alloc(std::size_t(m));
        chunkunits::gather_range(cv_sin, 0, std::size_t(m), t, arena);
        psin = t;
    }
    else
    {
        psin = cv_sin.ptr(0, 0);
    }
    if(cv_cos.nchunks > 1
        && !chunkunits::in_one_chunk(cv_cos, 0, std::size_t(m)))
    {
        T *t = arena.alloc(std::size_t(m));
        chunkunits::gather_range(cv_cos, 0, std::size_t(m), t, arena);
        pcos = t;
    }
    else
    {
        pcos = cv_cos.ptr(0, 0);
    }
    if(psin == nullptr || pcos == nullptr || !arena.ok_state())
    {
        return;
    }
    Index j = 0;
    while(j < n && arena.ok_state())
    {
        std::size_t const f = std::size_t(j) * col;
        bool const str_src = !chunkunits::in_one_chunk(cv_src, f, col);
        bool const str_dst = !chunkunits::in_one_chunk(cv_dst, f, col);
        if(str_src || str_dst)
        {
            // Straddling column on either side: stage the column whole
            // through the bounded temps.
            chunkunits::gather_range(cv_src, f, col, temp_src, arena);
            if(str_dst)
            {
                chunkunits::gather_range(cv_dst, f, col, temp_dst, arena);
            }
            if(!arena.ok_state())
            {
                return;
            }
            T *pd = str_dst ? temp_dst
                            : cv_dst.ptr(cv_dst.chunk_of(f),
                                  f - cv_dst.chunk_begin(cv_dst.chunk_of(f)));
            sub(m, Index{1}, psin, pcos, temp_src, pd);
            if(str_dst && arena.ok_state())
            {
                chunkunits::scatter_range(cv_dst, f, col, temp_dst, arena);
            }
            ++j;
            continue;
        }
        std::size_t const cs = cv_src.chunk_of(f);
        std::size_t const cd = cv_dst.chunk_of(f);
        Index Ls = static_cast<Index>((cv_src.chunk_end(cs) - f) / col);
        Index Ld = static_cast<Index>((cv_dst.chunk_end(cd) - f) / col);
        Index L = Ls < Ld ? Ls : Ld;
        if(L > n - j)
        {
            L = n - j;
        }
        if(L < 1)
        {
            L = 1;
        }
        sub(m, L, psin, pcos, cv_src.ptr(cs, f - cv_src.chunk_begin(cs)),
            cv_dst.ptr(cd, f - cv_dst.chunk_begin(cd)));
        j += L;
    }
}

} // namespace

//! Constructor
template<typename T>
Rope<std::tuple<T>>::Rope():
    codelet(
        "nntile_rope",
        &Rope<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Rope<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Rope<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::rope::cpu<T>
template<typename T>
void Rope<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_sin = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_cos = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[2]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[3]);
    if((buf_sin->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_cos->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
        && (buf_src->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_dst->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
       )
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::rope::cpu<T>(args->m, args->n, buf_sin->get_ptr<T>(),
            buf_cos->get_ptr<T>(), buf_src->get_ptr<T>(),
            buf_dst->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        rope_chunked<T>(
            args->m,
            args->n,
            buf_sin,
            buf_cos,
            buf_src,
            buf_dst,
            arena,
            [](Index mm, Index nn, T const *sn, T const *cc, T const *ps,
                T *pd) {
                kernel::rope::cpu<T>(mm, nn, sn, cc, ps, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_rope: cpu chunked driver aborted after an allocation "
            "failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void Rope<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Rope<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Rope<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Rope<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Rope<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Rope<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! NNHaul wrapper for kernel::rope::cuda<T>
template<typename T>
void Rope<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t*>(cl_args);
    // Get interfaces
    auto buf_sin = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_cos = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[2]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[3]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if((buf_sin->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_cos->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
        && (buf_src->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_dst->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
       )
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::rope::cuda<T>(stream, args->m, args->n, buf_sin->get_ptr<T>(),
            buf_cos->get_ptr<T>(), buf_src->get_ptr<T>(),
            buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ArenaCuda<T> arena(stream);
    rope_chunked<T>(
        args->m,
        args->n,
        buf_sin,
        buf_cos,
        buf_src,
        buf_dst,
        arena,
        [stream](Index mm, Index nn, T const *sn, T const *cc, T const *ps,
            T *pd) {
            kernel::rope::cuda<T>(stream, mm, nn, sn, cc, ps, pd);
        });
    if(!arena.ok_state())
    {
        std::fprintf(stderr,
            "nntile_rope: chunked CUDA driver aborted after a stream "
            "error; dst left unwritten\n");
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Rope<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Rope<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Rope<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Rope<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Rope<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Rope<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for rope tasks
template<typename T>
std::uint64_t Rope<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, and k
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    return hash;
}

template<typename T>
void Rope<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, ::nnhaul::Handle & sin, ::nnhaul::Handle & cos, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
//! Insert rope task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked column driver; rope never stages.
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &sin }, { STARPU_R, &cos }, { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Rope<std::tuple<nntile::fp64_t>>;
template class Rope<std::tuple<nntile::fp32_t>>;
template class Rope<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Rope<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Rope<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Rope<std::tuple<nntile::fp16_t>>;
template class Rope<std::tuple<nntile::bf16_t>>;

//! Pack of rope operations for different types
rope_pack_t rope;

} // namespace nntile::haul
