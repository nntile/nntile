/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/hypot.cc
 * hypot: direct kernel versus the NNHaul wrapper on the same inputs.
 *
 * @version 2.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/hypot.hh>
#include <nntile/nnhaul/ops/hypot.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index nelems)
{
    Scalar const alpha = 0.5;
    Scalar const beta = -1.5;
    auto src1 = wrap::series<T>(static_cast<std::size_t>(nelems), 1, 1);
    auto src2 = wrap::series<T>(static_cast<std::size_t>(nelems), 2, 1);
    auto dst = wrap::series<T>(static_cast<std::size_t>(nelems), -1, -1);
    auto cpu_dst = dst;
    std::cout << "Run kernel::hypot::cpu\n";
    nntile::kernel::hypot::cpu<T>(
        nelems, alpha, src1.data(), beta, src2.data(), cpu_dst.data());
    {
        ::nnhaul::Handle a(sizeof(T) * src1.size());
        ::nnhaul::Handle b(sizeof(T) * src2.size());
        ::nnhaul::Handle c(sizeof(T) * dst.size());
        wrap::store(a, src1);
        wrap::store(b, src2);
        wrap::store(c, dst);
        nntile::nnhaul::hypot.submit<std::tuple<T>>(
            wrap::cpu_worker, nelems, alpha, a, beta, b, c);
        wrap::expect_close("hypot cpu", cpu_dst, wrap::load<T>(c, dst.size()), 4);
    }
    wrap::Stream stream;
    wrap::Device<T> dev1(src1);
    wrap::Device<T> dev2(src2);
    wrap::Device<T> dev_dst(dst);
    std::cout << "Run kernel::hypot::cuda\n";
    nntile::kernel::hypot::cuda<T>(
        stream.stream, nelems, alpha, dev1.ptr, beta, dev2.ptr, dev_dst.ptr);
    auto gpu_dst = wrap::from_device(dev_dst.ptr, dst.size(), stream.stream);
    {
        ::nnhaul::Handle a(sizeof(T) * src1.size());
        ::nnhaul::Handle b(sizeof(T) * src2.size());
        ::nnhaul::Handle c(sizeof(T) * dst.size());
        wrap::store(a, src1);
        wrap::store(b, src2);
        wrap::store(c, dst);
        nntile::nnhaul::hypot.submit<std::tuple<T>>(
            wrap::cuda_worker, nelems, alpha, a, beta, b, c);
        wrap::expect_equal("hypot cuda", gpu_dst, wrap::load<T>(c, dst.size()));
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

            nntile::nnhaul::hypot.submit<std::tuple<T>>(
                wrap::cpu_worker, nelems, alpha, a, beta, b, c);
            wrap::expect_close(
                "hypot cpu chunked", cpu_dst,
                wrap::load<T>(c, dst.size()), 4);
        }
        {
            ::nnhaul::Handle a(sizeof(T) * src1.size(), chunk);
            ::nnhaul::Handle b(sizeof(T) * src2.size(), chunk);
            ::nnhaul::Handle c(sizeof(T) * dst.size(), chunk);

            wrap::store(a, src1);
            wrap::store(b, src2);
            wrap::store(c, dst);

            nntile::nnhaul::hypot.submit<std::tuple<T>>(
                wrap::cuda_worker, nelems, alpha, a, beta, b, c);
            wrap::expect_equal(
                "hypot cuda chunked", gpu_dst,
                wrap::load<T>(c, dst.size()));
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
