/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/maxsumexp.cc
 * maxsumexp: direct kernel versus the NNHaul wrapper on the same inputs.
 *
 * @version 2.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/maxsumexp.hh>
#include <nntile/nnhaul/ops/maxsumexp.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index m, Index n, Index k)
{
    using Y = typename T::repr_t;
    std::vector<T> src(static_cast<std::size_t>(m * n * k));
    std::vector<T> dst(static_cast<std::size_t>(2 * m * n));
    for (Index i = 0; i < m * n * k; ++i)
    {
        src[static_cast<std::size_t>(i)] = Y(i + 1);
    }
    for (Index i = 0; i < 2 * m * n; i += 2)
    {
        dst[static_cast<std::size_t>(i)] = Y(-i - 1);
        dst[static_cast<std::size_t>(i + 1)] = Y(2 * i);
    }
    auto cpu_dst = dst;
    std::cout << "Run kernel::maxsumexp::cpu\n";
    nntile::kernel::maxsumexp::cpu<T>(
        m, n, k, src.data(), Scalar{1.0}, cpu_dst.data());
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::maxsumexp.submit<std::tuple<T>>(
            wrap::cpu_worker, m, n, k, src_h, dst_h, Scalar{1.0});
        wrap::expect_equal(
            "maxsumexp cpu", cpu_dst, wrap::load<T>(dst_h, dst.size()));
    }
    wrap::Stream stream;
    wrap::Device<T> dev_src(src);
    wrap::Device<T> dev_dst(dst);
    std::cout << "Run kernel::maxsumexp::cuda\n";
    nntile::kernel::maxsumexp::cuda<T>(
        stream.stream, m, n, k, dev_src.ptr, Scalar{1.0}, dev_dst.ptr);
    auto gpu_dst = wrap::from_device(dev_dst.ptr, dst.size(), stream.stream);
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::maxsumexp.submit<std::tuple<T>>(
            wrap::cuda_worker, m, n, k, src_h, dst_h, Scalar{1.0});
        wrap::expect_equal(
            "maxsumexp cuda", gpu_dst, wrap::load<T>(dst_h, dst.size()));
    }
    // Chunked-handle variants (the NATIVE chunked unit driver: whole-row
    // pieces chained through the kernel's own beta!=0 accumulate merge -
    // algebraically exact but rounded differently from the single call,
    // so the big chunked legs compare with the norm-family tolerance
    // instead of bitwise; small legs stay unchunked and bitwise).
    for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * src.size()))
    {
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            wrap::Chunked dst_h(sizeof(T) * dst.size(), chunk);
            wrap::store(src_h.handle, src);
            wrap::store(dst_h.handle, dst);
            nntile::nnhaul::maxsumexp.submit<std::tuple<T>>(
                wrap::cpu_worker, m, n, k, src_h.handle, dst_h.handle,
                Scalar{1.0});
            wrap::expect_close(
                "maxsumexp cpu chunked", cpu_dst,
                wrap::load<T>(dst_h.handle, dst.size()), 16);
        }
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            wrap::Chunked dst_h(sizeof(T) * dst.size(), chunk);
            wrap::store(src_h.handle, src);
            wrap::store(dst_h.handle, dst);
            nntile::nnhaul::maxsumexp.submit<std::tuple<T>>(
                wrap::cuda_worker, m, n, k, src_h.handle, dst_h.handle,
                Scalar{1.0});
            wrap::expect_close(
                "maxsumexp cuda chunked", gpu_dst,
                wrap::load<T>(dst_h.handle, dst.size()), 16);
        }
    }
}

int main()
{
    wrap::Session session;
    (void)session;
    check<fp32_t>(3, 5, 7);
    check<fp64_t>(3, 5, 7);
    check<fp32_t>(100, 100, 20);
    return 0;
}
