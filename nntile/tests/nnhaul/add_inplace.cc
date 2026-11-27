/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/add_inplace.cc
 * add_inplace: direct kernel versus the NNHaul wrapper on the same inputs.
 *
 * @version 1.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/add_inplace.hh>
#include <nntile/nnhaul/ops/add_inplace.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index nelems, Scalar alpha, Scalar beta)
{
    auto src = wrap::series<T>(static_cast<std::size_t>(nelems), 1, 1);
    auto dst = wrap::series<T>(static_cast<std::size_t>(nelems), -2, 1);
    auto cpu_dst = dst;
    std::cout << "Run kernel::add_inplace::cpu\n";
    nntile::kernel::add_inplace::cpu<T>(
        nelems, alpha, src.data(), beta, cpu_dst.data());
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::add_inplace.submit<std::tuple<T>>(
            wrap::cpu_worker, nelems, alpha, src_h, beta, dst_h);
        wrap::expect_equal(
            "add_inplace cpu", cpu_dst, wrap::load<T>(dst_h, dst.size()));
    }
    wrap::Stream stream;
    wrap::Device<T> dev_src(src);
    wrap::Device<T> dev_dst(dst);
    std::cout << "Run kernel::add_inplace::cuda\n";
    nntile::kernel::add_inplace::cuda<T>(
        stream.stream, nelems, alpha, dev_src.ptr, beta, dev_dst.ptr);
    auto gpu_dst = wrap::from_device(dev_dst.ptr, dst.size(), stream.stream);
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::add_inplace.submit<std::tuple<T>>(
            wrap::cuda_worker, nelems, alpha, src_h, beta, dst_h);
        wrap::expect_equal(
            "add_inplace cuda", gpu_dst, wrap::load<T>(dst_h, dst.size()));
    }
    // Chunked-handle variants: odd small chunk plus the default policy
    // class, both compared against the same direct-kernel results.
    for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * src.size()))
    {
        {
            ::nnhaul::Handle src_h(sizeof(T) * src.size(), chunk);
            ::nnhaul::Handle dst_h(sizeof(T) * dst.size(), chunk);

            wrap::store(src_h, src);
            wrap::store(dst_h, dst);

            nntile::nnhaul::add_inplace.submit<std::tuple<T>>(
                wrap::cpu_worker, nelems, alpha, src_h, beta, dst_h);
            wrap::expect_equal(
                "add_inplace cpu chunked", cpu_dst,
                wrap::load<T>(dst_h, dst.size()));
        }
        {
            ::nnhaul::Handle src_h(sizeof(T) * src.size(), chunk);
            ::nnhaul::Handle dst_h(sizeof(T) * dst.size(), chunk);

            wrap::store(src_h, src);
            wrap::store(dst_h, dst);

            nntile::nnhaul::add_inplace.submit<std::tuple<T>>(
                wrap::cuda_worker, nelems, alpha, src_h, beta, dst_h);
            wrap::expect_equal(
                "add_inplace cuda chunked", gpu_dst,
                wrap::load<T>(dst_h, dst.size()));
        }
    }
}

int main()
{
    wrap::Session session;
    (void)session;
    check<fp32_t>(1, 0.5, -0.5);
    check<fp32_t>(10000, 1.0, 0.0);
    check<fp64_t>(1, 0.5, -0.5);
    check<fp64_t>(10000, 1.0, 0.0);
    check<fp32_t>(200000, 0.5, -0.5);
    return 0;
}
