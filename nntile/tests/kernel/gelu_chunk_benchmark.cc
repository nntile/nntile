/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file tests/kernel/gelu_chunk_benchmark.cc
 * Compare the one-launch chunked GeLU kernel with the old driver path
 * (one contiguous launch per chunk).
 *
 * Build: cmake --build <build> --target gelu_chunk_benchmark
 * Run:   ./gelu_chunk_benchmark [repeats]
 */

#include "nntile/kernel/gelu/cuda.hh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{

void check(cudaError_t err, char const *what)
{
    if(err != cudaSuccess)
    {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

template<typename F>
float time_ms(cudaStream_t stream, int repeats, F &&launch)
{
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create");
    check(cudaEventCreate(&stop), "event create");
    for(int i = 0; i < 3; ++i)
    {
        launch();
    }
    check(cudaStreamSynchronize(stream), "warmup sync");
    check(cudaEventRecord(start, stream), "event record");
    for(int i = 0; i < repeats; ++i)
    {
        launch();
    }
    check(cudaEventRecord(stop, stream), "event record");
    check(cudaStreamSynchronize(stream), "sync");
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / static_cast<float>(repeats);
}

} // namespace

int main(int argc, char **argv)
{
    int const repeats = argc > 1 ? std::atoi(argv[1]) : 20;
    if(repeats <= 0)
    {
        std::fprintf(stderr, "repeats must be positive\n");
        return 1;
    }
    // 1 MiB chunk class, float32.
    constexpr std::size_t kChunkBytes = std::size_t{1} << 20;
    constexpr std::size_t kChunkElems = kChunkBytes / sizeof(float);
    nntile::kernel::ChunkClass const cls{kChunkBytes - 1,
        static_cast<int>(__builtin_ctzll(kChunkBytes))};

    std::printf("%8s %8s %14s %14s %10s %12s %12s\n",
        "MiB", "chunks", "per-chunk ms", "chunked ms", "speedup",
        "per-chunk GB/s", "chunked GB/s");

    for(std::size_t mib = 1; mib <= 256; mib *= 2)
    {
        std::size_t const nchunks = mib; // 1 MiB class
        std::size_t const nbytes = mib << 20;
        std::size_t const nelems = nbytes / sizeof(float);

        float *base = nullptr;
        check(cudaMalloc(&base, nbytes), "cudaMalloc");
        check(cudaMemset(base, 0, nbytes), "cudaMemset");

        std::vector<const nntile::fp32_t *> h_src(nchunks);
        std::vector<nntile::fp32_t *> h_dst(nchunks);
        for(std::size_t c = 0; c < nchunks; ++c)
        {
            h_src[c] = reinterpret_cast<const nntile::fp32_t *>(
                base + c * kChunkElems);
            h_dst[c] = reinterpret_cast<nntile::fp32_t *>(
                base + c * kChunkElems);
        }
        const nntile::fp32_t **d_src = nullptr;
        nntile::fp32_t **d_dst = nullptr;
        check(cudaMalloc(&d_src, nchunks * sizeof(nntile::fp32_t *)), "cudaMalloc tab");
        check(cudaMalloc(&d_dst, nchunks * sizeof(nntile::fp32_t *)), "cudaMalloc tab");
        check(cudaMemcpy(d_src, h_src.data(), nchunks * sizeof(nntile::fp32_t *),
                cudaMemcpyHostToDevice), "cudaMemcpy tab");
        check(cudaMemcpy(d_dst, h_dst.data(), nchunks * sizeof(nntile::fp32_t *),
                cudaMemcpyHostToDevice), "cudaMemcpy tab");

        cudaStream_t stream = nullptr;
        check(cudaStreamCreate(&stream), "stream create");

        auto per_chunk = [&] {
            for(std::size_t c = 0; c < nchunks; ++c)
            {
                std::size_t const n = (c + 1 == nchunks)
                    ? nelems - c * kChunkElems : kChunkElems;
                nntile::kernel::gelu::cuda<nntile::fp32_t>(stream,
                    static_cast<nntile::Index>(n),
                    h_src[c], h_dst[c]);
            }
        };
        auto chunked = [&] {
            nntile::kernel::gelu::chunked<nntile::fp32_t>(stream,
                static_cast<nntile::Index>(nelems), d_src,
                static_cast<nntile::Index>(nchunks), d_dst,
                static_cast<nntile::Index>(nchunks),
                static_cast<nntile::Index>(kChunkElems),
                static_cast<nntile::Index>(kChunkElems), cls);
        };

        float const ms_old = time_ms(stream, repeats, per_chunk);
        float const ms_new = time_ms(stream, repeats, chunked);
        double const gb = static_cast<double>(nbytes) / 1e9;
        std::printf("%8zu %8zu %14.6f %14.6f %10.2f %12.1f %12.1f\n",
            mib, nchunks, ms_old, ms_new, ms_old / ms_new,
            gb / (ms_old / 1e3), gb / (ms_new / 1e3));

        check(cudaStreamDestroy(stream), "stream destroy");
        check(cudaFree(d_src), "cudaFree");
        check(cudaFree(d_dst), "cudaFree");
        check(cudaFree(base), "cudaFree");
    }
    return 0;
}
