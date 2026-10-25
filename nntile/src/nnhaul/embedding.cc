/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/embedding.cc
 * Embeddings from vocabulary within StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/embedding.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

// Other NNTile headers
#include "nntile/kernel/embedding.hh"
#include "nntile/kernel/nnhaul/chunk_embedding.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked embedding gather
//
// The kernel fills embed[i1, k_start+i0, i2] = vocab[index[i1, i2]*k_size +
// i0] for every (i1, i2) occurrence: the vocab side of each occurrence is a
// contiguous k_size-element row (the giant tied-embedding table), the embed
// side is a stride-m element run. A chunked vocab table therefore decomposes
// into per-(occurrence, vocab chunk segment) moves - a pure byte gather with
// no arithmetic - and a chunked embed output splits the same moves at the
// embed chunk boundaries of the strided run. All work happens inside the one
// task body (no extra tasks, no whole-tile scratch: the only host-side
// staging is the small m*n index list), and both buffers unchunked keeps the
// historical single kernel call, bitwise.
// ---------------------------------------------------------------------------

namespace
{

//! Chunk residency of one buffer's flat element space (equal-size chunks
//! with a shorter tail; an unchunked buffer is the 1-chunk degenerate case).
template<typename T>
struct EmbChunkView
{
    ::nnhaul::Buffer *buf = nullptr;
    std::size_t chunk_elems = 1;
    std::size_t nchunks = 1;
    std::size_t total_elems = 0;

    EmbChunkView() = default;
    explicit EmbChunkView(::nnhaul::Buffer *b):
        buf(b),
        total_elems(b->nbytes() / sizeof(T)),
        nchunks(b->chunk_count())
    {
        if(nchunks > 1)
        {
            chunk_elems = b->chunk_bytes(0) / sizeof(T);
        }
        else
        {
            chunk_elems = total_elems > 0 ? total_elems : 1;
            nchunks = 1;
        }
    }

    std::size_t chunk_of(std::size_t elem) const noexcept
    {
        return elem / chunk_elems;
    }
    std::size_t chunk_begin(std::size_t c) const noexcept
    {
        return c * chunk_elems;
    }
    std::size_t chunk_end(std::size_t c) const noexcept
    {
        std::size_t const e = (c + 1) * chunk_elems;
        return e < total_elems ? e : total_elems;
    }
    //! Typed pointer at chunk c, off elements into that chunk.
    T *ptr(std::size_t c, std::size_t off) const
    {
        if(nchunks > 1)
        {
            return buf->get_chunk_ptr<T>(c) + off;
        }
        return buf->get_ptr<T>() + off;
    }
};

//! Walk every (occurrence, vocab chunk segment) piece and invoke
//! copy_seg(src, dst_elem, dst_stride, count): src holds count contiguous
//! vocab elements that must land at embed flat elements dst_elem,
//! dst_elem+stride, ... (the kernel's strided embed run).
template<typename T, typename CopySeg>
void embedding_chunked(Index m, Index n, Index k, Index k_start,
    Index k_size, const int64_t *index, EmbChunkView<T> const &vocab,
    CopySeg &&copy_seg)
{
    std::size_t const row_len = static_cast<std::size_t>(k_size);
    for(Index i2 = 0; i2 < n; ++i2)
    {
        for(Index i1 = 0; i1 < m; ++i1)
        {
            std::size_t const row = row_len
                * static_cast<std::size_t>(static_cast<::int64_t>(
                    index[static_cast<std::size_t>(i2) * m + i1]));
            std::size_t done = 0;
            while(done < row_len)
            {
                std::size_t const at = row + done;
                std::size_t const c = vocab.chunk_of(at);
                std::size_t len = vocab.chunk_end(c) - at;
                if(len > row_len - done)
                {
                    len = row_len - done;
                }
                std::size_t const dst0 =
                    (static_cast<std::size_t>(i2) * k + k_start + done) * m
                    + i1;
                copy_seg(vocab.ptr(c, at - vocab.chunk_begin(c)), dst0,
                    static_cast<std::size_t>(m), len);
                done += len;
            }
        }
    }
}

//! Host-side tokens of the (possibly chunked) index buffer. The list is
//! m*n int64 values - bounded by the call shape, never tile-sized scratch.
template<typename HostCopy>
std::vector<int64_t> index_to_host(::nnhaul::Buffer *buf, Index m, Index n,
    HostCopy &&host_copy)
{
    std::size_t const nelems = static_cast<std::size_t>(m) * n;
    std::vector<int64_t> out(nelems == 0 ? 1 : nelems);
    if(buf->chunk_count() == 1)
    {
        host_copy(out.data(), buf->get_ptr<int64_t>(), nelems * sizeof(int64_t));
        return out;
    }
    std::size_t const chunk_elems = buf->chunk_bytes(0) / sizeof(int64_t);
    std::size_t done = 0;
    while(done < nelems)
    {
        std::size_t const c = done / chunk_elems;
        std::size_t const off = done - c * chunk_elems;
        std::size_t len = (buf->chunk_bytes(c) / sizeof(int64_t)) - off;
        if(len > nelems - done)
        {
            len = nelems - done;
        }
        host_copy(out.data() + done, buf->get_chunk_ptr<int64_t>(c) + off,
            len * sizeof(int64_t));
        done += len;
    }
    return out;
}

//! Mirror the kernel's loud token validation (kernels cannot throw; the
//! driver reads the tokens on the host, so it fails the same way before
//! touching any pointer offset derived from them).
void embedding_check_tokens(Index m, Index n, Index index_range,
    const int64_t *index) noexcept
{
    for(Index i2 = 0; i2 < n; ++i2)
    {
        for(Index i1 = 0; i1 < m; ++i1)
        {
            const ::int64_t token = static_cast<::int64_t>(
                index[static_cast<std::size_t>(i2) * m + i1]);
            if(token < 0 or token >= index_range)
            {
                std::fprintf(stderr,
                        "nntile embedding: token id %lld is outside the "
                        "vocabulary range [0, %lld)\n",
                        static_cast<long long>(token),
                        static_cast<long long>(index_range));
                std::abort();
            }
        }
    }
}

} // namespace

//! Constructor
template<typename T>
Embedding<std::tuple<T>>::Embedding():
    codelet(
        "nntile_embedding",
        &Embedding<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Embedding<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Embedding<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply embedding on StarPU buffer on CPU
template<typename T>
void Embedding<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_index = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_vocab = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_embed = static_cast<::nnhaul::Buffer *>(buffers[2]);
    if(buf_index->chunk_count() == 1 && buf_vocab->chunk_count() == 1
        && buf_embed->chunk_count() == 1)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        const int64_t *index = buf_index->get_ptr<int64_t>();
        const T *vocab = buf_vocab->get_ptr<T>();
        T *embed = buf_embed->get_ptr<T>();
        kernel::embedding::cpu<T>(
            args->m,
            args->n,
            args->k,
            args->k_start,
            args->k_size,
            args->index_range,
            index,
            vocab,
            embed
        );
        return;
    }
    // Gather the tokens host-side, validate them, then run the per-segment
    // moves against the chunk tables.
    std::vector<int64_t> const index = index_to_host(
        buf_index, args->m, args->n,
        [](void *dst, const void *src, std::size_t nbytes) {
            std::memcpy(dst, src, nbytes);
        });
    embedding_check_tokens(
        args->m, args->n, args->index_range, index.data());
    EmbChunkView<T> const vocab(buf_vocab);
    EmbChunkView<T> const embed(buf_embed);
    embedding_chunked<T>(
        args->m,
        args->n,
        args->k,
        args->k_start,
        args->k_size,
        index.data(),
        vocab,
        [&](const T *src, std::size_t dst0, std::size_t stride,
            std::size_t len) {
            for(std::size_t j = 0; j < len; ++j)
            {
                std::size_t const e = dst0 + j * stride;
                std::size_t const c = embed.chunk_of(e);
                embed.ptr(c, e - embed.chunk_begin(c))[0] = src[j];
            }
        });
}

// Specializations of CPU wrapper for accelerated types
template<>
void Embedding<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Embedding<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Embedding<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Embedding<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Embedding<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Embedding<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply embedding on StarPU buffer on CUDA
template<typename T>
void Embedding<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_index = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_vocab = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_embed = static_cast<::nnhaul::Buffer *>(buffers[2]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if(buf_index->chunk_count() == 1 && buf_vocab->chunk_count() == 1
        && buf_embed->chunk_count() == 1)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        const int64_t *index = buf_index->get_ptr<int64_t>();
        const T *vocab = buf_vocab->get_ptr<T>();
        T *embed = buf_embed->get_ptr<T>();
        kernel::embedding::cuda<T>(
            stream,
            args->m,
            args->n,
            args->k,
            args->k_start,
            args->k_size,
            args->index_range,
            index,
            vocab,
            embed
        );
        return;
    }
    // Fully asynchronous: the token index stays on the device; one
    // chunk-aware gather kernel (no D2H, no stream synchronization).
    ::nntile::kernel::nnhaul::embedding_gather(*buf_index, *buf_vocab, *buf_embed,
        static_cast<std::size_t>(args->m),
        static_cast<std::size_t>(args->n),
        static_cast<std::size_t>(args->k),
        static_cast<std::size_t>(args->k_start),
        static_cast<std::size_t>(args->k_size),
        static_cast<std::size_t>(args->index_range), sizeof(T));
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Embedding<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    Embedding<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Embedding<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    Embedding<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Embedding<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    Embedding<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for embedding tasks that depends only on cl_arg
template<typename T>
std::uint64_t Embedding<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, n, k and k_size.
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    hash = ::nntile::haul::fnv1a(&args->k_size, sizeof(args->k_size), hash);
    return hash;
}

template<typename T>
void Embedding<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, Index k_start, Index k_size,
        Index index_range,
        ::nnhaul::Handle & index, ::nnhaul::Handle & vocab, ::nnhaul::Handle & embed)
//! Insert embedding task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->k_start = k_start;
    args->k_size = k_size;
    args->index_range = index_range;
    double nflops = m * n * k_size;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked gather; embedding never goes through the staging fallback.
    (void)nflops;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &index }, { STARPU_R, &vocab }, { STARPU_RW, &embed } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Embedding<std::tuple<nntile::fp64_t>>;
template class Embedding<std::tuple<nntile::fp32_t>>;
template class Embedding<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Embedding<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Embedding<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Embedding<std::tuple<nntile::bf16_t>>;
template class Embedding<std::tuple<nntile::fp16_t>>;

//! Pack of embedding operations for different types
embedding_pack_t embedding;

} // namespace nntile::haul
