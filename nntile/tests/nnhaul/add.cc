/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/add.cc
 * add: direct kernel versus the NNHaul wrapper on the same inputs.
 *
 * @version 2.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/add.hh>
#include <nntile/nnhaul/ops/add.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index nelems, Scalar alpha, Scalar beta)
{
    auto src1 = wrap::series<T>(static_cast<std::size_t>(nelems), 1, 1);
    auto src2 = wrap::series<T>(static_cast<std::size_t>(nelems), -1, -1);
    auto dst = wrap::series<T>(static_cast<std::size_t>(nelems), 2, 2);
    auto cpu_dst = dst;
    std::cout << "Run kernel::add::cpu\n";
    nntile::kernel::add::cpu<T>(
        nelems, alpha, src1.data(), beta, src2.data(), cpu_dst.data());
    {
        ::nnhaul::Handle a(sizeof(T) * src1.size());
        ::nnhaul::Handle b(sizeof(T) * src2.size());
        ::nnhaul::Handle c(sizeof(T) * dst.size());
        wrap::store(a, src1);
        wrap::store(b, src2);
        wrap::store(c, dst);
        nntile::nnhaul::add.submit<std::tuple<T>>(
            wrap::cpu_worker, nelems, alpha, a, beta, b, c);
        wrap::expect_equal("add cpu", cpu_dst, wrap::load<T>(c, dst.size()));
    }
    wrap::Stream stream;
    wrap::Device<T> dev1(src1);
    wrap::Device<T> dev2(src2);
    wrap::Device<T> dev_dst(dst);
    std::cout << "Run kernel::add::cuda\n";
    nntile::kernel::add::cuda<T>(
        stream.stream, nelems, alpha, dev1.ptr, beta, dev2.ptr, dev_dst.ptr);
    auto gpu_dst = wrap::from_device(dev_dst.ptr, dst.size(), stream.stream);
    {
        ::nnhaul::Handle a(sizeof(T) * src1.size());
        ::nnhaul::Handle b(sizeof(T) * src2.size());
        ::nnhaul::Handle c(sizeof(T) * dst.size());
        wrap::store(a, src1);
        wrap::store(b, src2);
        wrap::store(c, dst);
        nntile::nnhaul::add.submit<std::tuple<T>>(
            wrap::cuda_worker, nelems, alpha, a, beta, b, c);
        wrap::expect_equal("add cuda", gpu_dst, wrap::load<T>(c, dst.size()));
    }
    // Chunked-handle variants: odd small chunk plus the default policy
    // class, both compared against the same direct-kernel results.
    for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * src1.size()))
    {
        {
            ::nnhaul::Handle a(sizeof(T) * src1.size(), chunk);
            ::nnhaul::Handle b(sizeof(T) * src2.size(), chunk);
            ::nnhaul::Handle c(sizeof(T) * dst.size(), chunk);

            wrap::store(a, src1);
            wrap::store(b, src2);
            wrap::store(c, dst);

            nntile::nnhaul::add.submit<std::tuple<T>>(
                wrap::cpu_worker, nelems, alpha, a, beta, b, c);
            wrap::expect_equal(
                "add cpu chunked", cpu_dst,
                wrap::load<T>(c, dst.size()));
        }
        {
            ::nnhaul::Handle a(sizeof(T) * src1.size(), chunk);
            ::nnhaul::Handle b(sizeof(T) * src2.size(), chunk);
            ::nnhaul::Handle c(sizeof(T) * dst.size(), chunk);

            wrap::store(a, src1);
            wrap::store(b, src2);
            wrap::store(c, dst);

            nntile::nnhaul::add.submit<std::tuple<T>>(
                wrap::cuda_worker, nelems, alpha, a, beta, b, c);
            wrap::expect_equal(
                "add cuda chunked", gpu_dst,
                wrap::load<T>(c, dst.size()));
        }
    }
}

int main()
{
    wrap::Session session;
    (void)session;
    check<fp32_t>(1, 1.0, -0.5);
    check<fp32_t>(10000, 1.5, 0.0);
    check<fp64_t>(1, 1.0, -0.5);
    check<fp64_t>(10000, 1.5, 0.0);
    check<fp32_t>(200000, 1.0, -0.5);
    return 0;
}
