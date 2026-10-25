/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/embedding_backward.cc
 * Embeddings from vocabulary within StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/embedding_backward.hh"

// Standard libraries
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

// Other NNTile headers
#include "nntile/kernel/embedding_backward.hh"
#include "nntile/kernel/nnhaul/chunk_embedding.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked embedding backward (arena scheme, cf. gemm.cc)
//
// The kernel accumulates vocab[index[i1,i2]*k_size + i0] += alpha *
// embed[i1, k_start+i0, i2]: the vocab side is the giant tied-embedding
// grad table (row-major, one k_size-element row per token), the embed side
// is the small m*k*n grad tile. A chunked vocab table is processed one
// chunk at a time through a BOUNDED temp (one chunk of rows plus one
// spare row), so no whole-table scratch ever exists:
//
//   * every vocab row belongs to the chunk that contains its FIRST
//     element; chunk c therefore owns the contiguous element range
//     [r0*k_size, r1*k_size) where r0 = ceil(chunk_begin/k_size) and
//     r1 = ceil(chunk_end/k_size) (a row straddling the chunk end drags
//     its tail along - the temp is the contiguous image of that whole
//     range, and the scatter-back rewrites the exact same range, whose
//     tail bytes live in the NEXT chunk's head. That is safe because the
//     entire loop runs inside one task body on one in-order stream, and
//     the next chunk's own image starts at r1*k_size - element ranges
//     never overlap even when byte ranges do);
//   * the temp holds one spare "garbage" row at index 0 followed by the
//     image of rows [r0, r1). The kernel sub-call keeps the full m*n
//     index space (the kernel derives the embed read position from
//     (i1, i2), so the index list cannot be compacted without breaking
//     the embed addressing); occurrences whose token row lives elsewhere
//     are pointed at the garbage row, accumulate harmlessly there and are
//     never scattered back. The cost is a bounded compute amplification
//     (one full m*n pass per vocab chunk hitting mostly the garbage row)
//     traded for zero new kernels;
//   * the embed operand is addressed by a single pointer, so a chunked
//     embed tile is gathered whole into an arena temp - bounded by the
//     small grad tile, never the giant table;
//   * beta==0 zeroes every vocab chunk up front (the plain kernel's
//     memset of vocab_nelems), then every sub-call accumulates onto the
//     gathered image with beta=1.
//
// Every sub-call is the EXISTING kernel with unchanged (m, n, k, k_start,
// k_size, alpha) and a row-shifted table view, so the CPU result is
// bitwise the plain call's (per row, the same contributions in the same
// (i2, i1) order); on CUDA the kernel accumulates with atomicAdd, whose
// order is unspecified even for the plain call, so duplicate indices may
// differ in the last ulp between chunked and plain runs. Both buffers
// unchunked keeps the historical single kernel call, bitwise.
// ---------------------------------------------------------------------------

namespace
{

//! Chunk residency of one buffer's flat element space (equal-size chunks
//! with a shorter tail; an unchunked buffer is the 1-chunk degenerate case).
template<typename T>
struct EmbBwdChunkView
{
    ::nnhaul::Buffer *buf = nullptr;
    std::size_t chunk_elems = 1;
    std::size_t nchunks = 1;
    std::size_t total_elems = 0;

    EmbBwdChunkView() = default;
    explicit EmbBwdChunkView(::nnhaul::Buffer *b):
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

//! Host-memory scratch arena for the CPU body.
struct EmbBwdArenaCpu
{
    std::vector<std::vector<unsigned char>> blocks;

    void *alloc(std::size_t bytes)
    {
        blocks.emplace_back(bytes == 0 ? 1 : bytes);
        return blocks.back().data();
    }
    template<typename T>
    T *alloc(std::size_t elems)
    {
        return static_cast<T *>(alloc(elems * sizeof(T)));
    }
    void copy_bytes(void *dst, void const *src, std::size_t bytes) noexcept
    {
        if(bytes > 0)
        {
            std::memcpy(dst, src, bytes);
        }
    }
    void zero_bytes(void *dst, std::size_t bytes) noexcept
    {
        if(bytes > 0)
        {
            std::memset(dst, 0, bytes);
        }
    }
};

#ifdef NNTILE_USE_CUDA
//! Stream-ordered device scratch arena for the CUDA body: every block is
//! cudaMallocAsync'd on the compute stream and freed (cudaFreeAsync) at
//! scope end, so all gather/kernel/scatter work enqueued in between is
//! stream-ordered before the release.
struct EmbBwdArenaCuda
{
    cudaStream_t stream;
    std::vector<void *> blocks;
    bool ok = true;

    explicit EmbBwdArenaCuda(cudaStream_t s): stream(s)
    {
    }
    void *alloc(std::size_t bytes)
    {
        void *p = nullptr;
        cudaError_t const err = cudaMallocAsync(&p, bytes == 0 ? 1 : bytes,
            stream);
        if(err != cudaSuccess)
        {
            // Same convention as the gemm chunked driver: report on
            // stderr; the codelet returns void, so a failed scratch
            // leaves the result partially unwritten (ok stops the rest).
            std::fprintf(stderr, "nntile_embedding_backward: "
                    "cudaMallocAsync failed: %s\n", cudaGetErrorString(err));
            ok = false;
            return nullptr;
        }
        blocks.push_back(p);
        return p;
    }
    template<typename T>
    T *alloc(std::size_t elems)
    {
        return static_cast<T *>(alloc(elems * sizeof(T)));
    }
    void copy_bytes(void *dst, void const *src, std::size_t bytes) noexcept
    {
        if(bytes == 0 || !ok)
        {
            return;
        }
        cudaError_t const err = cudaMemcpyAsync(dst, src, bytes,
            cudaMemcpyDeviceToDevice, stream);
        if(err != cudaSuccess)
        {
            std::fprintf(stderr, "nntile_embedding_backward: "
                    "cudaMemcpyAsync failed: %s\n", cudaGetErrorString(err));
            ok = false;
        }
    }
    void zero_bytes(void *dst, std::size_t bytes) noexcept
    {
        if(bytes == 0 || !ok)
        {
            return;
        }
        cudaError_t const err = cudaMemsetAsync(dst, 0, bytes, stream);
        if(err != cudaSuccess)
        {
            std::fprintf(stderr, "nntile_embedding_backward: "
                    "cudaMemsetAsync failed: %s\n", cudaGetErrorString(err));
            ok = false;
        }
    }
    void copy_h2d(void *dst, void const *src, std::size_t bytes) noexcept
    {
        if(bytes == 0 || !ok)
        {
            return;
        }
        // Pageable H2D: the call returns once the source has been staged,
        // so the caller's host buffer may die right after (documented
        // cudaMemcpyAsync behavior).
        cudaError_t const err = cudaMemcpyAsync(dst, src, bytes,
            cudaMemcpyHostToDevice, stream);
        if(err != cudaSuccess)
        {
            std::fprintf(stderr, "nntile_embedding_backward: host-to-device "
                    "copy failed: %s\n", cudaGetErrorString(err));
            ok = false;
        }
    }
    ~EmbBwdArenaCuda()
    {
        for(void *p : blocks)
        {
            cudaError_t const err = cudaFreeAsync(p, stream);
            if(err != cudaSuccess)
            {
                std::fprintf(stderr, "nntile_embedding_backward: "
                        "cudaFreeAsync failed: %s\n",
                        cudaGetErrorString(err));
            }
        }
    }
};
#endif // NNTILE_USE_CUDA

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

//! Mirror the kernel's loud token validation (the driver derives pointer
//! offsets from the tokens, so it must fail the same way first).
void embedding_backward_check_tokens(Index m, Index n, Index index_range,
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
                        "nntile embedding_backward: token id %lld is "
                        "outside the vocabulary range [0, %lld)\n",
                        static_cast<long long>(token),
                        static_cast<long long>(index_range));
                std::abort();
            }
        }
    }
}

//! Gather the flat element range [start, start+len) of a chunked buffer
//! into one contiguous scratch run (mirror: scatter it back).
template<typename T, typename Arena>
void emb_bwd_gather(EmbBwdChunkView<T> const &cv, std::size_t start,
    std::size_t len, T *dst, Arena &arena) noexcept
{
    std::size_t done = 0;
    while(done < len)
    {
        std::size_t const at = start + done;
        std::size_t const c = cv.chunk_of(at);
        std::size_t n = cv.chunk_end(c) - at;
        if(n > len - done)
        {
            n = len - done;
        }
        arena.copy_bytes(dst + done, cv.ptr(c, at - cv.chunk_begin(c)),
            n * sizeof(T));
        done += n;
    }
}

template<typename T, typename Arena>
void emb_bwd_scatter(EmbBwdChunkView<T> const &cv, std::size_t start,
    std::size_t len, T const *src, Arena &arena) noexcept
{
    std::size_t done = 0;
    while(done < len)
    {
        std::size_t const at = start + done;
        std::size_t const c = cv.chunk_of(at);
        std::size_t n = cv.chunk_end(c) - at;
        if(n > len - done)
        {
            n = len - done;
        }
        arena.copy_bytes(cv.ptr(c, at - cv.chunk_begin(c)), src + done,
            n * sizeof(T));
        done += n;
    }
}

} // namespace

//! Constructor
template<typename T>
EmbeddingBackward<std::tuple<T>>::EmbeddingBackward():
    codelet(
        "nntile_embedding_backward",
        &EmbeddingBackward<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &EmbeddingBackward<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &EmbeddingBackward<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply embedding backward on StarPU buffer on CPU
template<typename T>
void EmbeddingBackward<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_index = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_embed = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_vocab = static_cast<::nnhaul::Buffer *>(buffers[2]);
    if(buf_index->chunk_count() == 1 && buf_embed->chunk_count() == 1
        && buf_vocab->chunk_count() == 1)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::embedding_backward::cpu<T>(
            args->m,
            args->n,
            args->k,
            args->k_start,
            args->k_size,
            args->index_range,
            args->vocab_nelems,
            args->alpha,
            args->beta,
            buf_index->get_ptr<int64_t>(),
            buf_embed->get_ptr<T>(),
            buf_vocab->get_ptr<T>()
        );
        return;
    }
    std::vector<int64_t> index = index_to_host(
        buf_index, args->m, args->n,
        [](void *dst, const void *src, std::size_t nbytes) {
            std::memcpy(dst, src, nbytes);
        });
    embedding_backward_check_tokens(
        args->m, args->n, args->index_range, index.data());
    EmbBwdArenaCpu arena;
    // The chunked driver (see the block comment above).
    Index const k_size = args->k_size;
    std::size_t const row_len = static_cast<std::size_t>(k_size);
    EmbBwdChunkView<T> const vocab(buf_vocab);
    EmbBwdChunkView<T> const embed_view(buf_embed);
    const T *embed = nullptr;
    T *embed_scratch = nullptr;
    if(embed_view.nchunks > 1)
    {
        embed_scratch = arena.alloc<T>(embed_view.total_elems);
        emb_bwd_gather(embed_view, 0, embed_view.total_elems, embed_scratch,
            arena);
        embed = embed_scratch;
    }
    else
    {
        embed = embed_view.ptr(0, 0);
    }
    if(args->beta == Scalar{0.0})
    {
        for(std::size_t c = 0; c < vocab.nchunks; ++c)
        {
            arena.zero_bytes(vocab.ptr(c, 0),
                (vocab.chunk_end(c) - vocab.chunk_begin(c)) * sizeof(T));
        }
    }
    int64_t *temp_index = arena.alloc<int64_t>(
        static_cast<std::size_t>(args->m) * args->n);
    // One reusable temp: the garbage row plus the largest chunk's row
    // image (a fresh temp per chunk would add up to the whole table).
    std::size_t span_max = 0;
    for(std::size_t c = 0; c < vocab.nchunks; ++c)
    {
        std::size_t const begin = vocab.chunk_begin(c);
        std::size_t const end = vocab.chunk_end(c);
        std::size_t const r0 = (begin + row_len - 1) / row_len;
        std::size_t const r1_raw = (end + row_len - 1) / row_len;
        std::size_t const r1 = r1_raw < static_cast<std::size_t>(args->index_range)
            ? r1_raw
            : static_cast<std::size_t>(args->index_range);
        if(r0 < r1)
        {
            span_max = std::max(span_max, (r1 - r0) * row_len);
        }
    }
    if(span_max > 0)
    {
        T *temp = arena.alloc<T>(row_len + span_max);
        std::size_t const count = static_cast<std::size_t>(args->m) * args->n;
        for(std::size_t c = 0; c < vocab.nchunks; ++c)
        {
            // Rows whose FIRST element starts inside this chunk (a row
            // straddling the end drags its tail along, see above).
            std::size_t const begin = vocab.chunk_begin(c);
            std::size_t const end = vocab.chunk_end(c);
            std::size_t const r0 = (begin + row_len - 1) / row_len;
            std::size_t const r1_raw = (end + row_len - 1) / row_len;
            std::size_t const r1 = r1_raw < static_cast<std::size_t>(args->index_range)
                ? r1_raw
                : static_cast<std::size_t>(args->index_range);
            if(r0 >= r1)
            {
                continue;
            }
            std::size_t const span = (r1 - r0) * row_len;
            // The garbage row must not feed uninitialized bytes into the
            // kernel's read-modify-write accumulation.
            arena.zero_bytes(temp, row_len * sizeof(T));
            emb_bwd_gather(vocab, r0 * row_len, span, temp + row_len, arena);
            // Row-shifted index: garbage row 0 for out-of-chunk occurrences.
            for(std::size_t j = 0; j < count; ++j)
            {
                std::size_t const token = static_cast<std::size_t>(
                    static_cast<::int64_t>(index[j]));
                temp_index[j] = token >= r0 && token < r1
                    ? static_cast<int64_t>(token - r0 + 1)
                    : int64_t{0};
            }
            kernel::embedding_backward::cpu<T>(
                args->m,
                args->n,
                args->k,
                args->k_start,
                k_size,
                /*index_range=*/static_cast<Index>(r1 - r0 + 1),
                /*vocab_nelems=*/static_cast<Index>(row_len + span),
                args->alpha,
                /*beta=*/Scalar{1.0},
                temp_index,
                embed,
                temp
            );
            emb_bwd_scatter(vocab, r0 * row_len, span, temp + row_len, arena);
        }
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void EmbeddingBackward<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    EmbeddingBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void EmbeddingBackward<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    EmbeddingBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void EmbeddingBackward<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    EmbeddingBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply embedding backward on StarPU buffer on CUDA
template<typename T>
void EmbeddingBackward<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_index = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_embed = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_vocab = static_cast<::nnhaul::Buffer *>(buffers[2]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if(buf_index->chunk_count() == 1 && buf_embed->chunk_count() == 1
        && buf_vocab->chunk_count() == 1)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::embedding_backward::cuda<T>(
            stream,
            args->m,
            args->n,
            args->k,
            args->k_start,
            args->k_size,
            args->index_range,
            args->vocab_nelems,
            args->alpha,
            args->beta,
            buf_index->get_ptr<int64_t>(),
            buf_embed->get_ptr<T>(),
            buf_vocab->get_ptr<T>()
        );
        return;
    }
    // Fully asynchronous: the same atomic-accumulation kernel the
    // unchunked path uses, but addressing chunk tables directly; the
    // token index never leaves the device.
    ::nntile::kernel::nnhaul::embedding_backward_add(*buf_index, *buf_embed, *buf_vocab,
        static_cast<std::size_t>(args->m),
        static_cast<std::size_t>(args->n),
        static_cast<std::size_t>(args->k),
        static_cast<std::size_t>(args->k_start),
        static_cast<std::size_t>(args->k_size),
        static_cast<std::size_t>(args->index_range),
        static_cast<double>(args->alpha),
        static_cast<double>(args->beta), sizeof(T));
}

// Specializations of CUDA wrapper for accelerated types
template<>
void EmbeddingBackward<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    EmbeddingBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void EmbeddingBackward<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    EmbeddingBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void EmbeddingBackward<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    EmbeddingBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for embedding tasks that depends only on cl_arg
template<typename T>
std::uint64_t EmbeddingBackward<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, n, k and k_size.
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    hash = ::nntile::haul::fnv1a(&args->k_size, sizeof(args->k_size), hash);
    hash = ::nntile::haul::fnv1a(&args->index_range, sizeof(args->index_range), hash);
    hash = ::nntile::haul::fnv1a(&args->vocab_nelems, sizeof(args->vocab_nelems),
            hash);
    hash = ::nntile::haul::fnv1a(&args->alpha, sizeof(args->alpha), hash);
    hash = ::nntile::haul::fnv1a(&args->beta, sizeof(args->beta), hash);
    return hash;
}

template<typename T>
void EmbeddingBackward<std::tuple<T>>::submit(int starpu_worker_hint, Index m,
        Index n, Index k, Index k_start, Index k_size, Index index_range,
        Index vocab_nelems,
        Scalar alpha, Scalar beta, ::nnhaul::Handle & index, ::nnhaul::Handle & embed, ::nnhaul::Handle & vocab,
        int redux)
//! Insert embedding_backward task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    constexpr Scalar zero = 0, one = 1;
    // Access mode for the output vocab handle
    enum starpu_data_access_mode vocab_mode;
    if(beta == zero)
    {
        vocab_mode = STARPU_W;
    }
    else if(beta == one)
    {
        if(redux != 0)
        {
            vocab_mode = STARPU_REDUX;
        }
        else
        {
            vocab_mode = static_cast<starpu_data_access_mode>(
                    STARPU_RW | STARPU_COMMUTE);
        }
    }
    else
    {
        throw std::runtime_error("embedding_backward: beta must be 0.0 or 1.0");
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->k_start = k_start;
    args->k_size = k_size;
    args->index_range = index_range;
    args->vocab_nelems = vocab_nelems;
    args->alpha = alpha;
    args->beta = beta;
    double nflops = m * n * k_size;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked arena driver; embedding_backward never stages.
    (void)nflops;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &index }, { STARPU_R, &embed }, { vocab_mode, &vocab } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class EmbeddingBackward<std::tuple<nntile::fp64_t>>;
template class EmbeddingBackward<std::tuple<nntile::fp32_t>>;
template class EmbeddingBackward<std::tuple<nntile::fp32_fast_tf32_t>>;
template class EmbeddingBackward<std::tuple<nntile::fp32_fast_fp16_t>>;
template class EmbeddingBackward<std::tuple<nntile::fp32_fast_bf16_t>>;
template class EmbeddingBackward<std::tuple<nntile::bf16_t>>;
template class EmbeddingBackward<std::tuple<nntile::fp16_t>>;

//! Pack of embedding backward operations for different types
embedding_backward_pack_t embedding_backward;

} // namespace nntile::haul
