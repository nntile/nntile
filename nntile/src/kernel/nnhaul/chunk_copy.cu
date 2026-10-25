/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: a virtual planner
 * builds one plan per batch from a drained snapshot, and a literal
 * executor issues exactly the planned operations.
 *
 * @file src/kernel/nnhaul/chunk_copy.cu
 * Single-launch chunk gather/scatter: the chunk pointers travel as
 * kernel arguments (a by-value descriptor table, no host-to-device
 * upload, no per-chunk API call). Used by GEMM's workspace path.
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/nnhaul/chunk_copy.hh"

#include <nnhaul/nnhaul.hh>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace nntile::kernel::nnhaul
{

using ::nnhaul::Buffer;

namespace
{

// 128 chunks is far above any real operand (a 512 MiB operand at the
// 4 MiB class); overflow is a caller defect, loud.
constexpr int kMaxCopyChunks = 128;

struct ChunkCopyDesc
{
    void *dst[kMaxCopyChunks];
    void const *src[kMaxCopyChunks];
    unsigned long long len[kMaxCopyChunks];
    int n;
};

__global__ void chunk_copy_kernel(ChunkCopyDesc const desc)
{
    int const c = static_cast<int>(blockIdx.x);
    if (c >= desc.n)
    {
        return;
    }
    char *dst = static_cast<char *>(desc.dst[c]);
    char const *src = static_cast<char const *>(desc.src[c]);
    unsigned long long const len = desc.len[c];
    std::size_t const stride = static_cast<std::size_t>(blockDim.x) * 16u;
    for (unsigned long long off =
             static_cast<unsigned long long>(threadIdx.x) * 16u;
        off < len; off += stride)
    {
        if (off + 16u <= len)
        {
            uint4 const v = *reinterpret_cast<uint4 const *>(src + off);
            *reinterpret_cast<uint4 *>(dst + off) = v;
        }
        else
        {
            for (unsigned long long j = off; j < len; ++j)
            {
                dst[j] = src[j];
            }
        }
    }
}

void launch_copy(ChunkCopyDesc const &desc)
{
    if (desc.n == 0)
    {
        return;
    }
    if (desc.n > kMaxCopyChunks)
    {
        std::fprintf(stderr,
            "[nnhaul] chunk copy: %d chunks exceed the by-value limit "
            "(%d)\n",
            desc.n, kMaxCopyChunks);
        std::abort();
    }
    chunk_copy_kernel<<<static_cast<unsigned>(desc.n), 256, 0,
        ::nnhaul::cuda_stream()>>>(desc);
    cudaError_t const err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr,
            "[nnhaul] chunk copy: kernel launch failed: %s\n",
            cudaGetErrorString(err));
        std::abort();
    }
}

} // namespace

void gather_chunks(Buffer const &src, void *dst_contiguous)
{
    (void)::nnhaul::cuda_stream();
    std::size_t const n = src.chunk_count();
    if (n > static_cast<std::size_t>(kMaxCopyChunks))
    {
        std::fprintf(stderr,
            "[nnhaul] gather_chunks: %zu chunks exceed %d\n", n,
            kMaxCopyChunks);
        std::abort();
    }
    ChunkCopyDesc desc{};
    desc.n = static_cast<int>(n);
    char *dst = static_cast<char *>(dst_contiguous);
    std::size_t off = 0;
    for (std::size_t c = 0; c < n; ++c)
    {
        std::size_t const len = src.chunk_bytes(c);
        desc.dst[c] = dst + off;
        desc.src[c] = src.get_chunk_ptr<char>(c);
        desc.len[c] = static_cast<unsigned long long>(len);
        off += len;
    }
    launch_copy(desc);
}

void scatter_chunks(void const *src_contiguous, Buffer &dst)
{
    (void)::nnhaul::cuda_stream();
    std::size_t const n = dst.chunk_count();
    if (n > static_cast<std::size_t>(kMaxCopyChunks))
    {
        std::fprintf(stderr,
            "[nnhaul] scatter_chunks: %zu chunks exceed %d\n", n,
            kMaxCopyChunks);
        std::abort();
    }
    ChunkCopyDesc desc{};
    desc.n = static_cast<int>(n);
    char const *src = static_cast<char const *>(src_contiguous);
    std::size_t off = 0;
    for (std::size_t c = 0; c < n; ++c)
    {
        std::size_t const len = dst.chunk_bytes(c);
        desc.dst[c] = dst.get_chunk_ptr<char>(c);
        desc.src[c] = src + off;
        desc.len[c] = static_cast<unsigned long long>(len);
        off += len;
    }
    launch_copy(desc);
}

} // namespace nntile::kernel::nnhaul
