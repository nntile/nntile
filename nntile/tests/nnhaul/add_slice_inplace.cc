/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/add_slice_inplace.cc
 * add_slice_inplace: direct kernel versus the NNHaul wrapper on the same
 * inputs.
 *
 * @version 1.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/add_slice_inplace.hh>
#include <nntile/nnhaul/ops/add_slice_inplace.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index m, Index n, Index k)
{
    using Y = typename T::repr_t;
    std::vector<T> src(static_cast<std::size_t>(m * n));
    std::vector<T> dst(static_cast<std::size_t>(m * n * k));
    for (Index i = 0; i < m * n; ++i)
    {
        src[static_cast<std::size_t>(i)] = Y(2 * i + 2);
    }
    for (Index i = 0; i < m * n * k; ++i)
    {
        dst[static_cast<std::size_t>(i)] = Y(-i - 1);
    }
    auto cpu_dst = dst;
    std::cout << "Run kernel::add_slice_inplace::cpu\n";
    nntile::kernel::add_slice_inplace::cpu<T>(
        m, n, k, 0.5, src.data(), -0.5, cpu_dst.data());
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::add_slice_inplace.submit<std::tuple<T>>(
            wrap::cpu_worker, m, n, k, 0.5, src_h, -0.5, dst_h);
        wrap::expect_equal(
            "add_slice_inplace cpu", cpu_dst, wrap::load<T>(dst_h, dst.size()));
    }
    wrap::Stream stream;
    wrap::Device<T> dev_src(src);
    wrap::Device<T> dev_dst(dst);
    std::cout << "Run kernel::add_slice_inplace::cuda\n";
    nntile::kernel::add_slice_inplace::cuda<T>(
        stream.stream, m, n, k, 0.5, dev_src.ptr, -0.5, dev_dst.ptr);
    auto gpu_dst = wrap::from_device(dev_dst.ptr, dst.size(), stream.stream);
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::add_slice_inplace.submit<std::tuple<T>>(
            wrap::cuda_worker, m, n, k, 0.5, src_h, -0.5, dst_h);
        wrap::expect_equal(
            "add_slice_inplace cuda", gpu_dst, wrap::load<T>(dst_h, dst.size()));
    }
    // Chunked-handle variants (staged fallback; RW dst gathers and
    // scatters), same references.
    for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * dst.size()))
    {
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            wrap::Chunked dst_h(sizeof(T) * dst.size(), chunk);
            wrap::store(src_h.handle, src);
            wrap::store(dst_h.handle, dst);
            nntile::nnhaul::add_slice_inplace.submit<std::tuple<T>>(
                wrap::cpu_worker, m, n, k, 0.5, src_h.handle, -0.5,
                dst_h.handle);
            wrap::expect_equal(
                "add_slice_inplace cpu chunked", cpu_dst,
                wrap::load<T>(dst_h.handle, dst.size()));
        }
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            wrap::Chunked dst_h(sizeof(T) * dst.size(), chunk);
            wrap::store(src_h.handle, src);
            wrap::store(dst_h.handle, dst);
            nntile::nnhaul::add_slice_inplace.submit<std::tuple<T>>(
                wrap::cuda_worker, m, n, k, 0.5, src_h.handle, -0.5,
                dst_h.handle);
            wrap::expect_equal(
                "add_slice_inplace cuda chunked", gpu_dst,
                wrap::load<T>(dst_h.handle, dst.size()));
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
