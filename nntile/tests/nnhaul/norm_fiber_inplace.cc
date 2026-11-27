/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/norm_fiber_inplace.cc
 * norm_fiber_inplace: direct kernel versus the NNHaul wrapper on the same
 * inputs.
 *
 * @version 1.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/norm_fiber_inplace.hh>
#include <nntile/nnhaul/ops/norm_fiber_inplace.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index m, Index n, Index k, Index batch, Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    std::vector<T> src(static_cast<std::size_t>(m * n * k * batch));
    std::vector<T> dst(static_cast<std::size_t>(k * batch));
    for (Index b = 0; b < batch; ++b)
    {
        for (Index i2 = 0; i2 < k; ++i2)
        {
            dst[static_cast<std::size_t>(b * batch + i2)] = Y{0.0};
            for (Index i1 = 0; i1 < n; ++i1)
            {
                T *slice = src.data() + ((i1 + b * n) * k + i2) * m;
                for (Index i0 = 0; i0 < m; ++i0)
                {
                    slice[i0] = Y{-1.0};
                }
            }
        }
    }
    auto cpu_dst = dst;
    std::cout << "Run kernel::norm_fiber_inplace::cpu\n";
    nntile::kernel::norm_fiber_inplace::cpu<T>(
        m, n, k, batch, alpha, src.data(), beta, cpu_dst.data());
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::norm_fiber_inplace.submit<std::tuple<T>>(
            wrap::cpu_worker, m, n, k, batch, alpha, src_h, beta, dst_h, 0);
        wrap::expect_close(
            "norm_fiber_inplace cpu",
cpu_dst,
            wrap::load<T>(dst_h, dst.size()), 16);
    }
    wrap::Stream stream;
    wrap::Device<T> dev_src(src);
    wrap::Device<T> dev_dst(dst);
    std::cout << "Run kernel::norm_fiber_inplace::cuda\n";
    nntile::kernel::norm_fiber_inplace::cuda<T>(
        stream.stream, m, n, k, batch, alpha, dev_src.ptr, beta, dev_dst.ptr);
    auto gpu_dst = wrap::from_device(dev_dst.ptr, dst.size(), stream.stream);
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::norm_fiber_inplace.submit<std::tuple<T>>(
            wrap::cuda_worker, m, n, k, batch, alpha, src_h, beta, dst_h, 0);
        wrap::expect_equal(
            "norm_fiber_inplace cuda",
            gpu_dst,
            wrap::load<T>(dst_h, dst.size()));
    }
    // Chunked-handle variants (the NATIVE chunked fiber driver:
    // whole-fiber pieces chained through the kernel's piecewise hypot
    // merge - algebraically exact but rounded differently from the
    // single call, so the chunked legs compare with the norm-family
    // tolerance; the RW dst staging temps are gone).
    for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * src.size()))
    {
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            wrap::Chunked dst_h(sizeof(T) * dst.size(), chunk);
            wrap::store(src_h.handle, src);
            wrap::store(dst_h.handle, dst);
            nntile::nnhaul::norm_fiber_inplace.submit<std::tuple<T>>(
                wrap::cpu_worker, m, n, k, batch, alpha, src_h.handle, beta,
                dst_h.handle, 0);
            wrap::expect_close(
                "norm_fiber_inplace cpu chunked",
                cpu_dst,
                wrap::load<T>(dst_h.handle, dst.size()),
                16);
        }
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            wrap::Chunked dst_h(sizeof(T) * dst.size(), chunk);
            wrap::store(src_h.handle, src);
            wrap::store(dst_h.handle, dst);
            nntile::nnhaul::norm_fiber_inplace.submit<std::tuple<T>>(
                wrap::cuda_worker, m, n, k, batch, alpha, src_h.handle, beta,
                dst_h.handle, 0);
            wrap::expect_close(
                "norm_fiber_inplace cuda chunked",
                gpu_dst,
                wrap::load<T>(dst_h.handle, dst.size()),
                16);
        }
    }
}

int main()
{
    wrap::Session session;
    (void)session;
    check<fp64_t>(32, 32, 10, 1, 1.0, 0.0);
    check<fp64_t>(32, 9, 10, 1, 1.0, 0.0);
    check<fp32_t>(32, 32, 10, 1, 1.0, 0.0);
    check<fp32_t>(32, 9, 10, 1, 1.0, 0.0);
    check<fp32_t>(32, 32, 40, 1, 1.0, 0.0);
    return 0;
}
