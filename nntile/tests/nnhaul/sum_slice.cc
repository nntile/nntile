/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/sum_slice.cc
 * sum_slice: direct kernel versus the NNHaul wrapper on the same inputs.
 *
 * @version 2.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/sum_slice.hh>
#include <nntile/nnhaul/ops/sum_slice.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index m, Index n, Index k, Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    std::vector<T> src(static_cast<std::size_t>(m * n * k));
    std::vector<T> dst(static_cast<std::size_t>(m * n));
    for (Index i = 0; i < m * n * k; ++i)
    {
        src[static_cast<std::size_t>(i)] = Y(i + 1);
    }
    for (Index i = 0; i < m * n; ++i)
    {
        dst[static_cast<std::size_t>(i)] = Y(-i - 1);
    }
    auto cpu_dst = dst;
    std::cout << "Run kernel::sum_slice::cpu\n";
    nntile::kernel::sum_slice::cpu<T>(
        m, n, k, alpha, src.data(), beta, cpu_dst.data());
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::sum_slice.submit<std::tuple<T>>(
            wrap::cpu_worker, m, n, k, alpha, src_h, beta, dst_h);
        wrap::expect_equal(
            "sum_slice cpu", cpu_dst, wrap::load<T>(dst_h, dst.size()));
    }
    wrap::Stream stream;
    wrap::Device<T> dev_src(src);
    wrap::Device<T> dev_dst(dst);
    std::cout << "Run kernel::sum_slice::cuda\n";
    nntile::kernel::sum_slice::cuda<T>(
        stream.stream, m, n, k, alpha, dev_src.ptr, beta, dev_dst.ptr);
    auto gpu_dst = wrap::from_device(dev_dst.ptr, dst.size(), stream.stream);
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::sum_slice.submit<std::tuple<T>>(
            wrap::cuda_worker, m, n, k, alpha, src_h, beta, dst_h);
        wrap::expect_equal(
            "sum_slice cuda", gpu_dst, wrap::load<T>(dst_h, dst.size()));
    }
    // Chunked-handle variants (the NATIVE chunked unit driver:
    // whole-row pieces chained through the kernel's own beta!=0
    // accumulate merge - algebraically exact but rounded differently
    // from the single call, so the big chunked legs compare with the
    // norm-family tolerance instead of bitwise; small legs stay
    // unchunked and bitwise).
    for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * src.size()))
    {
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            wrap::Chunked dst_h(sizeof(T) * dst.size(), chunk);
            wrap::store(src_h.handle, src);
            wrap::store(dst_h.handle, dst);
            nntile::nnhaul::sum_slice.submit<std::tuple<T>>(
                wrap::cpu_worker, m, n, k, alpha, src_h.handle, beta,
                dst_h.handle);
            wrap::expect_close(
                "sum_slice cpu chunked", cpu_dst,
                wrap::load<T>(dst_h.handle, dst.size()), 16);
        }
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            wrap::Chunked dst_h(sizeof(T) * dst.size(), chunk);
            wrap::store(src_h.handle, src);
            wrap::store(dst_h.handle, dst);
            nntile::nnhaul::sum_slice.submit<std::tuple<T>>(
                wrap::cuda_worker, m, n, k, alpha, src_h.handle, beta,
                dst_h.handle);
            wrap::expect_close(
                "sum_slice cuda chunked", gpu_dst,
                wrap::load<T>(dst_h.handle, dst.size()), 16);
        }
    }
}

int main()
{
    wrap::Session session;
    (void)session;
    check<fp32_t>(3, 5, 7, 1.0, -1.0);
    check<fp32_t>(3, 5, 7, 2.0, 0.0);
    check<fp32_t>(3, 5, 7, 0.0, 1.0);
    check<fp64_t>(3, 5, 7, 1.0, -1.0);
    check<fp64_t>(3, 5, 7, 2.0, 0.0);
    check<fp64_t>(3, 5, 7, 0.0, 1.0);
    check<fp32_t>(100, 100, 20, 1.0, -1.0);
    return 0;
}
