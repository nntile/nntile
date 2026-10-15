/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/accumulate_maxsumexp.cc
 * Accumulate one StarPU maxsumexp buffer into another
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/accumulate_maxsumexp.hh"

// Standard libraries
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/accumulate_maxsumexp.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked accumulate_maxsumexp driver (chunk_units.hh scheme).
//
// Kernel index map (derived from kernel::accumulate_maxsumexp::cpu/cuda,
// NOT assumed): src and dst are interleaved (max, sumexp) PAIR arrays -
// element (2*i, 2*i+1) - and the op merges pair i of src into pair i of
// dst independent of every other pair. The natural unit is therefore
// the 2-element pair: pieces are runs of whole pairs inside single src
// AND dst chunks (the (max, sumexp) pair is never split - the kernel's
// merge reads both halves), a straddling pair is gathered into a
// bounded 2-element temp (both sides: the kernel reads src AND dst),
// merged as a 1-pair sub-call and scattered back. Every pair is
// deterministic in isolation, so the piecewise result is bitwise the
// plain call. Zero new kernels; all-unchunked buffers keep the single
// classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void accumulate_maxsumexp_chunked(Index npairs,
    ::nnhaul::Buffer *buf_src, ::nnhaul::Buffer *buf_dst,
    std::size_t force_bytes, Arena &arena, Sub &&sub)
{
    if(npairs <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_src(buf_src, force_bytes);
    chunkunits::ChunkView<T> const cv_dst(buf_dst, force_bytes);
    bool const chunked_src = cv_src.nchunks > 1 || force_bytes != 0;
    bool const chunked_dst = cv_dst.nchunks > 1 || force_bytes != 0;
    T *temp_src = chunked_src ? arena.alloc(std::size_t(2)) : nullptr;
    T *temp_dst = chunked_dst ? arena.alloc(std::size_t(2)) : nullptr;
    Index done = 0;
    while(done < npairs && arena.ok_state())
    {
        std::size_t const f = std::size_t(2) * done;
        bool const src_str = !chunkunits::in_one_chunk(cv_src, f, 2);
        bool const dst_str = !chunkunits::in_one_chunk(cv_dst, f, 2);
        if(src_str || dst_str)
        {
            // Straddling pair: stage both halves (the kernel reads src
            // AND dst), merge one pair, scatter dst back.
            chunkunits::gather_range(cv_src, f, 2, temp_src, arena);
            chunkunits::gather_range(cv_dst, f, 2, temp_dst, arena);
            if(arena.ok_state())
            {
                sub(Index{1}, temp_src, temp_dst);
                chunkunits::scatter_range(cv_dst, f, 2, temp_dst, arena);
            }
            ++done;
            continue;
        }
        // Run of whole pairs to the nearer chunk end of either side.
        std::size_t const c_s = cv_src.chunk_of(f);
        std::size_t const c_d = cv_dst.chunk_of(f);
        std::size_t n = cv_src.chunk_end(c_s);
        n = n < cv_dst.chunk_end(c_d) ? n : cv_dst.chunk_end(c_d);
        n = n - f;
        Index run = static_cast<Index>(n / 2);
        if(run > npairs - done)
        {
            run = npairs - done;
        }
        if(run <= 0)
        {
            run = 1;
        }
        sub(run, cv_src.ptr(c_s, f - cv_src.chunk_begin(c_s)),
            cv_dst.ptr(c_d, f - cv_dst.chunk_begin(c_d)));
        done += run;
    }
}

} // namespace

//! Constructor
template<typename T>
AccumulateMaxSumExp<std::tuple<T>>::AccumulateMaxSumExp():
    codelet(
        "nntile_accumulate_maxsumexp",
        &AccumulateMaxSumExp<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &AccumulateMaxSumExp<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        nullptr)
{
    // Modes cannot be variable for accumulate_maxsumexp operation
    // Construct modes
    constexpr std::array<starpu_data_access_mode, 2> modes = {
        static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE),
        STARPU_R
    };
    // Set modes
    codelet.set_modes_fixed(modes);
}

//! Apply accumulate_maxsumexp operation for StarPU buffers in CPU
template<typename T>
void AccumulateMaxSumExp<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get interfaces
    Index npairs =
        (*reinterpret_cast<std::size_t const *>(cl_args)) / sizeof(T) / 2;
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[1]);
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    if(buf_src->chunk_count() == 1 && buf_dst->chunk_count() == 1
        && forced == 0)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::accumulate_maxsumexp::cpu<T>(npairs, buf_src->get_ptr<T>(),
            buf_dst->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        accumulate_maxsumexp_chunked<T>(npairs, buf_src, buf_dst, forced,
            arena,
            [](Index n, T const *s, T *d) {
                kernel::accumulate_maxsumexp::cpu<T>(n, s, d);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_accumulate_maxsumexp: cpu chunked driver aborted "
            "after an allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void AccumulateMaxSumExp<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateMaxSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AccumulateMaxSumExp<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateMaxSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AccumulateMaxSumExp<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateMaxSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


#ifdef NNTILE_USE_CUDA
//! Apply accumulate_maxsumexp for StarPU buffers on CUDA
template<typename T>
void AccumulateMaxSumExp<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get interfaces
    Index npairs =
        (*reinterpret_cast<std::size_t const *>(cl_args)) / sizeof(T) / 2;
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[1]);
    cudaStream_t stream = ::nnhaul::cuda_stream();
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    if(buf_src->chunk_count() == 1 && buf_dst->chunk_count() == 1
        && forced == 0)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::accumulate_maxsumexp::cuda<T>(stream, npairs,
            buf_src->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ArenaCuda<T> arena(stream);
    accumulate_maxsumexp_chunked<T>(npairs, buf_src, buf_dst, forced, arena,
        [stream](Index n, T const *s, T *d) {
            kernel::accumulate_maxsumexp::cuda<T>(stream, n, s, d);
        });
    if(!arena.ok_state())
    {
        std::fprintf(stderr,
            "nntile_accumulate_maxsumexp: chunked CUDA driver aborted "
            "after a stream error; dst left unwritten\n");
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void AccumulateMaxSumExp<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateMaxSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AccumulateMaxSumExp<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateMaxSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AccumulateMaxSumExp<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateMaxSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

template<typename T>
void AccumulateMaxSumExp<std::tuple<T>>::submit(int starpu_worker_hint, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
//! Insert accumulate_maxsumexp task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    //double nflops;
    // Submit task
    std::size_t nnhaul_nbytes = dst.nbytes();
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked pair-unit driver; accumulate_maxsumexp never stages.
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_RW | STARPU_COMMUTE, &dst }, { STARPU_R, &src } }, &nnhaul_nbytes, sizeof(nnhaul_nbytes));

}


// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AccumulateMaxSumExp<std::tuple<nntile::fp64_t>>;
template class AccumulateMaxSumExp<std::tuple<nntile::fp32_t>>;
template class AccumulateMaxSumExp<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AccumulateMaxSumExp<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AccumulateMaxSumExp<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AccumulateMaxSumExp<std::tuple<nntile::bf16_t>>;

//! Pack of accumulate_maxsumexp operations for different types
accumulate_maxsumexp_pack_t accumulate_maxsumexp;

} // namespace nntile::haul
