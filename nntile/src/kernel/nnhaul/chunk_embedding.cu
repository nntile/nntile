/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: a virtual planner
 * builds one plan per batch from a drained snapshot, and a literal
 * executor issues exactly the planned operations.
 *
 * @file src/kernel/nnhaul/chunk_embedding.cu
 * Device-side embedding gather/scatter over chunk tables: the token
 * index stays on the device, so the codelet needs no D2H and no
 * cudaStreamSynchronize. One launch per operation.
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/nnhaul/chunk_embedding.hh"

#include <nnhaul/nnhaul.hh>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace nntile::kernel::nnhaul
{

using ::nnhaul::Buffer;
using ::nnhaul::kChunkClassBytes;

namespace
{

constexpr int kMaxTableChunks = 128;

// One tile's chunk table, by value in the kernel arguments.
struct ChunkTable
{
    void *ptr[kMaxTableChunks];
    unsigned long long chunk_elems; // elements in a full chunk
    int n;
};

constexpr int log2_u64(unsigned long long v)
{
    int s = 0;
    while ((1ull << s) < v)
    {
        ++s;
    }
    return s;
}

// The chunk class is the compile-time kChunkClassBytes and a power of
// two, and every supported element size is a power of two, so a flat
// position maps to (chunk, offset) with a shift/mask - no 64-bit
// division on the GPU address path.
template <typename T> struct ChunkGeom
{
    static constexpr unsigned long long elems =
        static_cast<unsigned long long>(kChunkClassBytes) / sizeof(T);
    static constexpr int shift = log2_u64(elems);
    static constexpr unsigned long long mask = elems - 1;
};

template <typename T>
__device__ __forceinline__ T *chunk_at(ChunkTable const &t,
    unsigned long long pos)
{
    unsigned long long const c = pos >> ChunkGeom<T>::shift;
    unsigned long long const off = pos & ChunkGeom<T>::mask;
    return static_cast<T *>(t.ptr[c]) + off;
}

__device__ void check_token(long long token, unsigned long long range,
    char const *who)
{
    if (token < 0 || token >= static_cast<long long>(range))
    {
        printf("nntile %s: token id %lld is outside the vocabulary "
               "range [0, %llu)\n",
            who, token, range);
        __trap();
    }
}

// embed[i0, k_start + i2, i1] = vocab[k_size*index[i0 + m*i1] + i2]
template <typename T>
__global__ void embedding_gather_kernel(unsigned long long m,
    unsigned long long n, unsigned long long k,
    unsigned long long k_start, unsigned long long k_size,
    unsigned long long index_range, int64_t const *index,
    ChunkTable vocab, ChunkTable embed, unsigned long long total)
{
    unsigned long long flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= total)
    {
        return;
    }
    unsigned long long const i0 = flat % m;
    unsigned long long const t = flat / m;
    unsigned long long const i2 = t % k_size;
    unsigned long long const i1 = t / k_size;
    long long const token = static_cast<long long>(index[i0 + m * i1]);
    check_token(token, index_range, "embedding");
    T const v = *chunk_at<T>(vocab,
        static_cast<unsigned long long>(token) * k_size + i2);
    *chunk_at<T>(embed, (i1 * k + k_start + i2) * m + i0) = v;
}

// vocab[k_size*index[..] + i2] += alpha * embed[(i1*k+k_start+i2)*m+i0]
template <typename T>
__global__ void embedding_backward_kernel(unsigned long long m,
    unsigned long long n, unsigned long long k,
    unsigned long long k_start, unsigned long long k_size,
    unsigned long long index_range, double alpha, int64_t const *index,
    ChunkTable embed, ChunkTable vocab, unsigned long long total)
{
    unsigned long long flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= total)
    {
        return;
    }
    unsigned long long const i0 = flat % m;
    unsigned long long const t = flat / m;
    unsigned long long const i2 = t % k_size;
    unsigned long long const i1 = t / k_size;
    long long const token = static_cast<long long>(index[i0 + m * i1]);
    check_token(token, index_range, "embedding_backward");
    T const v = *chunk_at<T>(embed, (i1 * k + k_start + i2) * m + i0);
    T *dst = chunk_at<T>(vocab,
        static_cast<unsigned long long>(token) * k_size + i2);
    atomicAdd(dst, static_cast<T>(alpha * static_cast<double>(v)));
}

ChunkTable table_of(Buffer const &buf, std::size_t elem_size)
{
    ChunkTable t{};
    std::size_t const nch = buf.chunk_count();
    if (nch > static_cast<std::size_t>(kMaxTableChunks))
    {
        std::fprintf(stderr,
            "[nnhaul] embedding: %zu chunks exceed the by-value table "
            "(%d)\n",
            nch, kMaxTableChunks);
        std::abort();
    }
    t.n = static_cast<int>(nch);
    t.chunk_elems = kChunkClassBytes / elem_size;
    for (std::size_t c = 0; c < nch; ++c)
    {
        t.ptr[c] = buf.get_chunk_ptr<char>(c);
    }
    return t;
}

int64_t const *index_ptr(Buffer const &index)
{
    if (index.chunk_count() != 1)
    {
        std::fprintf(stderr,
            "[nnhaul] embedding: the token index must be one chunk "
            "(got %zu)\n", index.chunk_count());
        std::abort();
    }
    return index.get_chunk_ptr<int64_t>(0);
}

} // namespace

void embedding_gather(Buffer const &index, Buffer const &vocab,
    Buffer &embed, std::size_t m, std::size_t n, std::size_t k,
    std::size_t k_start, std::size_t k_size, std::size_t index_range,
    std::size_t elem_size)
{
    (void)::nnhaul::cuda_stream();
    unsigned long long const total =
        static_cast<unsigned long long>(m) * n * k_size;
    if (total == 0)
    {
        return;
    }
    ChunkTable const vt = table_of(vocab, elem_size);
    ChunkTable const et = table_of(embed, elem_size);
    unsigned int const threads = 256;
    unsigned long long const blocks = (total + threads - 1) / threads;
    if (elem_size == 4)
    {
        embedding_gather_kernel<float><<<static_cast<unsigned>(blocks),
            threads, 0, ::nnhaul::cuda_stream()>>>(m, n, k, k_start, k_size,
            index_range, index_ptr(index), vt, et, total);
    }
    else if (elem_size == 8)
    {
        embedding_gather_kernel<double><<<static_cast<unsigned>(blocks),
            threads, 0, ::nnhaul::cuda_stream()>>>(m, n, k, k_start, k_size,
            index_range, index_ptr(index), vt, et, total);
    }
    else
    {
        std::fprintf(stderr,
            "[nnhaul] embedding: unsupported element size %zu\n",
            elem_size);
        std::abort();
    }
    cudaError_t const err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr,
            "[nnhaul] embedding gather launch failed: %s\n",
            cudaGetErrorString(err));
        std::abort();
    }
}

void embedding_backward_add(Buffer const &index, Buffer const &embed,
    Buffer &vocab, std::size_t m, std::size_t n, std::size_t k,
    std::size_t k_start, std::size_t k_size, std::size_t index_range,
    double alpha, double beta, std::size_t elem_size)
{
    (void)::nnhaul::cuda_stream();
    ChunkTable const vt = table_of(vocab, elem_size);
    ChunkTable const et = table_of(embed, elem_size);
    if (beta == 0.0)
    {
        for (std::size_t c = 0; c < vocab.chunk_count(); ++c)
        {
            cudaError_t const err = cudaMemsetAsync(
                vocab.get_chunk_ptr<char>(c), 0, vocab.chunk_bytes(c),
                ::nnhaul::cuda_stream());
            if (err != cudaSuccess)
            {
                std::fprintf(stderr,
                    "[nnhaul] embedding backward zero failed: %s\n",
                    cudaGetErrorString(err));
                std::abort();
            }
        }
    }
    unsigned long long const total =
        static_cast<unsigned long long>(m) * n * k_size;
    if (total == 0)
    {
        return;
    }
    unsigned int const threads = 256;
    unsigned long long const blocks = (total + threads - 1) / threads;
    if (elem_size == 4)
    {
        embedding_backward_kernel<float><<<static_cast<unsigned>(blocks),
            threads, 0, ::nnhaul::cuda_stream()>>>(m, n, k, k_start, k_size,
            index_range, alpha, index_ptr(index), et, vt, total);
    }
    else if (elem_size == 8)
    {
        embedding_backward_kernel<double><<<static_cast<unsigned>(blocks),
            threads, 0, ::nnhaul::cuda_stream()>>>(m, n, k, k_start, k_size,
            index_range, alpha, index_ptr(index), et, vt, total);
    }
    else
    {
        std::fprintf(stderr,
            "[nnhaul] embedding backward: unsupported element size %zu\n",
            elem_size);
        std::abort();
    }
    cudaError_t const err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr,
            "[nnhaul] embedding backward launch failed: %s\n",
            cudaGetErrorString(err));
        std::abort();
    }
}

} // namespace nntile::kernel::nnhaul
