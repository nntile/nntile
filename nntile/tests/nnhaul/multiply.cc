/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/multiply.cc
 * multiply: direct kernel versus the NNHaul wrapper on the same inputs.
 *
 * @version 1.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/multiply.hh>
#include <nntile/nnhaul/ops/multiply.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index nelems)
{
    using Y = typename T::repr_t;
    std::vector<T> src1(static_cast<std::size_t>(nelems));
    std::vector<T> src2(static_cast<std::size_t>(nelems));
    std::vector<T> dst(static_cast<std::size_t>(nelems));
    for (Index i = 0; i < nelems; ++i)
    {
        src1[static_cast<std::size_t>(i)] = Y(2 * i + 1 - nelems) / Y{1000};
        src2[static_cast<std::size_t>(i)] = Y(nelems - i) / Y{1000};
        dst[static_cast<std::size_t>(i)] = Y(-i - 1);
    }
    auto cpu_dst = dst;
    std::cout << "Run kernel::multiply::cpu\n";
    nntile::kernel::multiply::cpu<T>(
        nelems, 1.0, src1.data(), src2.data(), cpu_dst.data());
    {
        ::nnhaul::Handle a(sizeof(T) * src1.size());
        ::nnhaul::Handle b(sizeof(T) * src2.size());
        ::nnhaul::Handle c(sizeof(T) * dst.size());
        wrap::store(a, src1);
        wrap::store(b, src2);
        wrap::store(c, dst);
        nntile::nnhaul::multiply.submit<std::tuple<T>>(
            wrap::cpu_worker, nelems, 1.0, a, b, c);
        wrap::expect_equal("multiply cpu", cpu_dst, wrap::load<T>(c, dst.size()));
    }
    wrap::Stream stream;
    wrap::Device<T> dev1(src1);
    wrap::Device<T> dev2(src2);
    wrap::Device<T> dev_dst(dst);
    std::cout << "Run kernel::multiply::cuda\n";
    nntile::kernel::multiply::cuda<T>(
        stream.stream, nelems, 1.0, dev1.ptr, dev2.ptr, dev_dst.ptr);
    auto gpu_dst = wrap::from_device(dev_dst.ptr, dst.size(), stream.stream);
    {
        ::nnhaul::Handle a(sizeof(T) * src1.size());
        ::nnhaul::Handle b(sizeof(T) * src2.size());
        ::nnhaul::Handle c(sizeof(T) * dst.size());
        wrap::store(a, src1);
        wrap::store(b, src2);
        wrap::store(c, dst);
        nntile::nnhaul::multiply.submit<std::tuple<T>>(
            wrap::cuda_worker, nelems, 1.0, a, b, c);
        wrap::expect_equal("multiply cuda", gpu_dst, wrap::load<T>(c, dst.size()));
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

            nntile::nnhaul::multiply.submit<std::tuple<T>>(
                wrap::cpu_worker, nelems, 1.0, a, b, c);
            wrap::expect_equal(
                "multiply cpu chunked", cpu_dst,
                wrap::load<T>(c, dst.size()));
        }
        {
            ::nnhaul::Handle a(sizeof(T) * src1.size(), chunk);
            ::nnhaul::Handle b(sizeof(T) * src2.size(), chunk);
            ::nnhaul::Handle c(sizeof(T) * dst.size(), chunk);

            wrap::store(a, src1);
            wrap::store(b, src2);
            wrap::store(c, dst);

            nntile::nnhaul::multiply.submit<std::tuple<T>>(
                wrap::cuda_worker, nelems, 1.0, a, b, c);
            wrap::expect_equal(
                "multiply cuda chunked", gpu_dst,
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
