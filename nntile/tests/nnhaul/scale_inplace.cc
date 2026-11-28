/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/scale_inplace.cc
 * scale_inplace: direct kernel versus the NNHaul wrapper on the same
 * inputs.
 *
 * @version 2.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/scale_inplace.hh>
#include <nntile/nnhaul/ops/scale_inplace.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index nelems)
{
    Scalar const alpha = -0.5;
    auto init = wrap::series<T>(static_cast<std::size_t>(nelems), 1, 1);
    auto cpu_out = init;
    std::cout << "Run kernel::scale_inplace::cpu\n";
    nntile::kernel::scale_inplace::cpu<T>(nelems, alpha, cpu_out.data());
    {
        ::nnhaul::Handle data(sizeof(T) * init.size());
        wrap::store(data, init);
        nntile::nnhaul::scale_inplace.submit<std::tuple<T>>(
            wrap::cpu_worker, nelems, alpha, data);
        wrap::expect_equal(
            "scale_inplace cpu", cpu_out, wrap::load<T>(data, init.size()));
    }
    wrap::Stream stream;
    wrap::Device<T> dev(init);
    std::cout << "Run kernel::scale_inplace::cuda\n";
    nntile::kernel::scale_inplace::cuda<T>(stream.stream, nelems, alpha, dev.ptr);
    auto gpu_out = wrap::from_device(dev.ptr, init.size(), stream.stream);
    {
        ::nnhaul::Handle data(sizeof(T) * init.size());
        wrap::store(data, init);
        nntile::nnhaul::scale_inplace.submit<std::tuple<T>>(
            wrap::cuda_worker, nelems, alpha, data);
        wrap::expect_equal(
            "scale_inplace cuda", gpu_out, wrap::load<T>(data, init.size()));
    }
    // Chunked-handle variants: odd small chunk plus the default policy
    // class, both compared against the same direct-kernel results.
    for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * init.size()))
    {
        {
            ::nnhaul::Handle data(sizeof(T) * init.size(), chunk);

            wrap::store(data, init);

            nntile::nnhaul::scale_inplace.submit<std::tuple<T>>(
                wrap::cpu_worker, nelems, alpha, data);
            wrap::expect_equal(
                "scale_inplace cpu chunked", cpu_out,
                wrap::load<T>(data, init.size()));
        }
        {
            ::nnhaul::Handle data(sizeof(T) * init.size(), chunk);

            wrap::store(data, init);

            nntile::nnhaul::scale_inplace.submit<std::tuple<T>>(
                wrap::cuda_worker, nelems, alpha, data);
            wrap::expect_equal(
                "scale_inplace cuda chunked", gpu_out,
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
