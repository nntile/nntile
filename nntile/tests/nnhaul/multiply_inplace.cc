/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/multiply_inplace.cc
 * multiply_inplace: direct kernel versus the NNHaul wrapper on the same
 * inputs.
 *
 * @version 1.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/multiply_inplace.hh>
#include <nntile/nnhaul/ops/multiply_inplace.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index nelems)
{
    using Y = typename T::repr_t;
    std::vector<T> src(static_cast<std::size_t>(nelems));
    std::vector<T> dst(static_cast<std::size_t>(nelems));
    for (Index i = 0; i < nelems; ++i)
    {
        src[static_cast<std::size_t>(i)] = Y(2 * i + 1 - nelems) / Y{1000};
        dst[static_cast<std::size_t>(i)] = Y(nelems - i) / Y{1000};
    }
    auto cpu_dst = dst;
    std::cout << "Run kernel::multiply_inplace::cpu\n";
    nntile::kernel::multiply_inplace::cpu<T>(
        nelems, 1.0, src.data(), cpu_dst.data());
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::multiply_inplace.submit<std::tuple<T>>(
            wrap::cpu_worker, nelems, 1.0, src_h, dst_h);
        wrap::expect_equal(
            "multiply_inplace cpu", cpu_dst, wrap::load<T>(dst_h, dst.size()));
    }
    wrap::Stream stream;
    wrap::Device<T> dev_src(src);
    wrap::Device<T> dev_dst(dst);
    std::cout << "Run kernel::multiply_inplace::cuda\n";
    nntile::kernel::multiply_inplace::cuda<T>(
        stream.stream, nelems, 1.0, dev_src.ptr, dev_dst.ptr);
    auto gpu_dst = wrap::from_device(dev_dst.ptr, dst.size(), stream.stream);
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::multiply_inplace.submit<std::tuple<T>>(
            wrap::cuda_worker, nelems, 1.0, src_h, dst_h);
        wrap::expect_equal(
            "multiply_inplace cuda", gpu_dst, wrap::load<T>(dst_h, dst.size()));
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

            nntile::nnhaul::multiply_inplace.submit<std::tuple<T>>(
                wrap::cpu_worker, nelems, 1.0, src_h, dst_h);
            wrap::expect_equal(
                "multiply_inplace cpu chunked", cpu_dst,
                wrap::load<T>(dst_h, dst.size()));
        }
        {
            ::nnhaul::Handle src_h(sizeof(T) * src.size(), chunk);
            ::nnhaul::Handle dst_h(sizeof(T) * dst.size(), chunk);

            wrap::store(src_h, src);
            wrap::store(dst_h, dst);

            nntile::nnhaul::multiply_inplace.submit<std::tuple<T>>(
                wrap::cuda_worker, nelems, 1.0, src_h, dst_h);
            wrap::expect_equal(
                "multiply_inplace cuda chunked", gpu_dst,
                wrap::load<T>(dst_h, dst.size()));
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
