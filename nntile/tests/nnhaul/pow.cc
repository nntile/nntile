/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/pow.cc
 * pow: direct kernel versus the NNHaul wrapper on the same inputs.
 *
 * @version 2.0.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#include "harness.hh"

#include <tuple>

#include <nntile/kernel/pow.hh>
#include <nntile/nnhaul/ops/pow.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index nelems)
{
    Scalar const alpha = 1.0;
    Scalar const exp = 2.0;
    auto init = wrap::series<T>(static_cast<std::size_t>(nelems), 1, 1);
    auto cpu_out = init;
    std::cout << "Run kernel::pow::cpu\n";
    nntile::kernel::pow::cpu<T>(nelems, alpha, exp, cpu_out.data());
    {
        ::nnhaul::Handle data(sizeof(T) * init.size());
        wrap::store(data, init);
        nntile::nnhaul::pow.submit<std::tuple<T>>(
            wrap::cpu_worker, nelems, alpha, exp, data);
        wrap::expect_close("pow cpu", cpu_out, wrap::load<T>(data, init.size()), 4);
    }
    wrap::Stream stream;
    wrap::Device<T> dev(init);
    std::cout << "Run kernel::pow::cuda\n";
    nntile::kernel::pow::cuda<T>(stream.stream, nelems, alpha, exp, dev.ptr);
    auto gpu_out = wrap::from_device(dev.ptr, init.size(), stream.stream);
    {
        ::nnhaul::Handle data(sizeof(T) * init.size());
        wrap::store(data, init);
        nntile::nnhaul::pow.submit<std::tuple<T>>(
            wrap::cuda_worker, nelems, alpha, exp, data);
        wrap::expect_equal("pow cuda", gpu_out, wrap::load<T>(data, init.size()));
    }
    // Chunked-handle variants: odd small chunk plus the default policy
    // class, both compared against the same direct-kernel results.
    for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * init.size()))
    {
        {
            ::nnhaul::Handle data(sizeof(T) * init.size(), chunk);

            wrap::store(data, init);

            nntile::nnhaul::pow.submit<std::tuple<T>>(
                wrap::cpu_worker, nelems, alpha, exp, data);
            wrap::expect_close(
                "pow cpu chunked", cpu_out,
                wrap::load<T>(data, init.size()), 4);
        }
        {
            ::nnhaul::Handle data(sizeof(T) * init.size(), chunk);

            wrap::store(data, init);

            nntile::nnhaul::pow.submit<std::tuple<T>>(
                wrap::cuda_worker, nelems, alpha, exp, data);
            wrap::expect_equal(
                "pow cuda chunked", gpu_out,
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
