/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/fill.cc
 * fill: direct kernel versus the NNHaul wrapper on the same inputs.
 *
 * @version 2.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/fill.hh>
#include <nntile/nnhaul/ops/fill.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index nelems)
{
    Scalar const value = -0.5;
    auto init = wrap::series<T>(static_cast<std::size_t>(nelems), 1, 1);
    auto cpu_out = init;
    std::cout << "Run kernel::fill::cpu\n";
    nntile::kernel::fill::cpu<T>(nelems, value, cpu_out.data());
    {
        ::nnhaul::Handle data(sizeof(T) * init.size());
        wrap::store(data, init);
        nntile::nnhaul::fill.submit<std::tuple<T>>(
            wrap::cpu_worker, nelems, value, data);
        wrap::expect_equal(
            "fill cpu", cpu_out, wrap::load<T>(data, init.size()));
    }
    wrap::Stream stream;
    wrap::Device<T> dev(init);
    std::cout << "Run kernel::fill::cuda\n";
    nntile::kernel::fill::cuda<T>(stream.stream, nelems, value, dev.ptr);
    auto gpu_out = wrap::from_device(dev.ptr, init.size(), stream.stream);
    {
        ::nnhaul::Handle data(sizeof(T) * init.size());
        wrap::store(data, init);
        nntile::nnhaul::fill.submit<std::tuple<T>>(
            wrap::cuda_worker, nelems, value, data);
        wrap::expect_equal(
            "fill cuda", gpu_out, wrap::load<T>(data, init.size()));
    }
    // Chunked-handle variants: odd small chunk plus the default policy
    // class, both compared against the same direct-kernel results.
    for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * init.size()))
    {
        {
            ::nnhaul::Handle data(sizeof(T) * init.size(), chunk);

            wrap::store(data, init);

            nntile::nnhaul::fill.submit<std::tuple<T>>(
                wrap::cpu_worker, nelems, value, data);
            wrap::expect_equal(
                "fill cpu chunked", cpu_out,
                wrap::load<T>(data, init.size()));
        }
        {
            ::nnhaul::Handle data(sizeof(T) * init.size(), chunk);

            wrap::store(data, init);

            nntile::nnhaul::fill.submit<std::tuple<T>>(
                wrap::cuda_worker, nelems, value, data);
            wrap::expect_equal(
                "fill cuda chunked", gpu_out,
                wrap::load<T>(data, init.size()));
        }
    }
}

int main()
{
    wrap::Session session;
    (void)session;
    check<fp32_t>(1);
    check<fp32_t>(10000);
    check<fp64_t>(1);
    check<fp64_t>(10000);
    check<fp32_t>(200000);
    return 0;
}
