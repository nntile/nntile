/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/clear.cc
 * clear: direct kernel versus the NNHaul wrapper on the same inputs.
 *
 * @version 2.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/nnhaul/ops/clear.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

void check(std::size_t nbytes)
{
    std::vector<unsigned char> init(nbytes, 0xff);
    std::vector<unsigned char> cpu_out = init;
    std::cout << "Run memset\n";
    std::memset(cpu_out.data(), 0, nbytes);
    {
        ::nnhaul::Handle data(nbytes);
        wrap::store(data, init);
        nntile::nnhaul::clear.submit(wrap::cpu_worker, data);
        wrap::expect_bytes(
            "clear cpu", cpu_out, wrap::load<unsigned char>(data, nbytes));
    }
    wrap::Stream stream;
    unsigned char *dev = nullptr;
    wrap::cuda_ok(cudaMalloc(&dev, nbytes));
    wrap::cuda_ok(cudaMemcpy(
        dev, init.data(), nbytes, cudaMemcpyHostToDevice));
    std::cout << "Run cudaMemsetAsync\n";
    wrap::cuda_ok(cudaMemsetAsync(dev, 0, nbytes, stream.stream));
    auto gpu_out = wrap::from_device(dev, nbytes, stream.stream);
    wrap::cuda_ok(cudaFree(dev));
    {
        ::nnhaul::Handle data(nbytes);
        wrap::store(data, init);
        nntile::nnhaul::clear.submit(wrap::cuda_worker, data);
        wrap::expect_bytes(
            "clear cuda", gpu_out, wrap::load<unsigned char>(data, nbytes));
    }
    // Chunked-handle variants: odd small chunk plus the default policy
    // class, both equal to the same memset result.
    for (std::size_t chunk : wrap::chunk_variants(nbytes))
    {
        {
            ::nnhaul::Handle data(nbytes, chunk);
            wrap::store(data, init);
            nntile::nnhaul::clear.submit(wrap::cpu_worker, data);
            wrap::expect_bytes(
                "clear cpu chunked", cpu_out,
                wrap::load<unsigned char>(data, nbytes));
        }
        {
            ::nnhaul::Handle data(nbytes, chunk);
            wrap::store(data, init);
            nntile::nnhaul::clear.submit(wrap::cuda_worker, data);
            wrap::expect_bytes(
                "clear cuda chunked", gpu_out,
                wrap::load<unsigned char>(data, nbytes));
        }
    }
}

int main()
{
    wrap::Session session;
    (void)session;
    check(1);
    check(100000);
    check(2000000);
    return 0;
}
