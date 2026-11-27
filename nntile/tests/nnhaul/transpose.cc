/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/transpose.cc
 * transpose: direct kernel versus the NNHaul wrapper on the same inputs.
 *
 * @version 1.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/transpose.hh>
#include <nntile/nnhaul/ops/transpose.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index m, Index n)
{
    using Y = typename T::repr_t;
    std::vector<T> src(static_cast<std::size_t>(m * n));
    std::vector<T> dst(static_cast<std::size_t>(m * n));
    for (Index i = 0; i < m * n; ++i)
    {
        src[static_cast<std::size_t>(i)] = Y(2 * i + 2);
        dst[static_cast<std::size_t>(i)] = Y(-i - 1);
    }
    auto cpu_dst = dst;
    std::cout << "Run kernel::transpose::cpu\n";
    nntile::kernel::transpose::cpu<T>(m, n, 0.5, src.data(), cpu_dst.data());
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::transpose.submit<std::tuple<T>>(
            wrap::cpu_worker, m, n, 0.5, src_h, dst_h);
        wrap::expect_equal(
            "transpose cpu", cpu_dst, wrap::load<T>(dst_h, dst.size()));
    }
    wrap::Stream stream;
    wrap::Device<T> dev_src(src);
    wrap::Device<T> dev_dst(dst);
    std::cout << "Run kernel::transpose::cuda\n";
    nntile::kernel::transpose::cuda<T>(
        stream.stream, m, n, 0.5, dev_src.ptr, dev_dst.ptr);
    auto gpu_dst = wrap::from_device(dev_dst.ptr, dst.size(), stream.stream);
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::transpose.submit<std::tuple<T>>(
            wrap::cuda_worker, m, n, 0.5, src_h, dst_h);
        wrap::expect_equal(
            "transpose cuda", gpu_dst, wrap::load<T>(dst_h, dst.size()));
    }
    // Chunked-handle variants: odd small chunk plus the default policy
    // class, through the NATIVE chunked panel driver (family 2) and
    // compared against the same direct-kernel results; the degenerate
    // torture geometry (chunk <= one dst row) exercises the submit-time
    // staging-fallback guard in staged_ops.cc.
    for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * src.size()))
    {
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            wrap::Chunked dst_h(sizeof(T) * dst.size(), chunk);
            wrap::store(src_h.handle, src);
            wrap::store(dst_h.handle, dst);
            nntile::nnhaul::transpose.submit<std::tuple<T>>(
                wrap::cpu_worker, m, n, 0.5, src_h.handle, dst_h.handle);
            wrap::expect_equal(
                "transpose cpu chunked", cpu_dst,
                wrap::load<T>(dst_h.handle, dst.size()));
        }
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            wrap::Chunked dst_h(sizeof(T) * dst.size(), chunk);
            wrap::store(src_h.handle, src);
            wrap::store(dst_h.handle, dst);
            nntile::nnhaul::transpose.submit<std::tuple<T>>(
                wrap::cuda_worker, m, n, 0.5, src_h.handle, dst_h.handle);
            wrap::expect_equal(
                "transpose cuda chunked", gpu_dst,
                wrap::load<T>(dst_h.handle, dst.size()));
        }
    }
}

int main()
{
    wrap::Session session;
    (void)session;
    check<fp32_t>(3, 5);
    check<fp64_t>(3, 5);
    check<fp32_t>(520, 320);
    // Multi-chunk (> one 4 MiB class) rectangular transposes: the
    // chunked driver path with tail chunks and straddling tiles.
    check<fp32_t>(2048, 1024);
    check<fp32_t>(1234, 2345);
    return 0;
}
