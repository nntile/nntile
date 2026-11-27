/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/norm_fiber.cc
 * norm_fiber: direct kernel versus the NNHaul wrapper on the same inputs.
 *
 * @version 1.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/norm_fiber.hh>
#include <nntile/nnhaul/ops/norm_fiber.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index m, Index n, Index k, Index batch, Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    std::vector<T> src1(static_cast<std::size_t>(m * n * k * batch));
    std::vector<T> src2(static_cast<std::size_t>(k * batch));
    std::vector<T> dst(static_cast<std::size_t>(k * batch));
    for (Index b = 0; b < batch; ++b)
    {
        for (Index i2 = 0; i2 < k; ++i2)
        {
            dst[static_cast<std::size_t>(b * batch + i2)] = Y{1.0};
            src2[static_cast<std::size_t>(b * batch + i2)] = Y{1.0};
            for (Index i1 = 0; i1 < n; ++i1)
            {
                T *slice = src1.data() + ((i1 + b * n) * k + i2) * m;
                for (Index i0 = 0; i0 < m; ++i0)
                {
                    slice[i0] = Y{-1.0};
                }
            }
        }
    }
    auto cpu_dst = dst;
    std::cout << "Run kernel::norm_fiber::cpu\n";
    nntile::kernel::norm_fiber::cpu<T>(
        m, n, k, batch, alpha, src1.data(), beta, src2.data(), cpu_dst.data());
    {
        ::nnhaul::Handle a(sizeof(T) * src1.size());
        ::nnhaul::Handle b(sizeof(T) * src2.size());
        ::nnhaul::Handle c(sizeof(T) * dst.size());
        wrap::store(a, src1);
        wrap::store(b, src2);
        wrap::store(c, dst);
        nntile::nnhaul::norm_fiber.submit<std::tuple<T>>(
            wrap::cpu_worker, m, n, k, batch, alpha, a, beta, b, c, 0);
        wrap::expect_close(
            "norm_fiber cpu", cpu_dst, wrap::load<T>(c, dst.size()), 16);
    }
    wrap::Stream stream;
    wrap::Device<T> dev1(src1);
    wrap::Device<T> dev2(src2);
    wrap::Device<T> dev_dst(dst);
    std::cout << "Run kernel::norm_fiber::cuda\n";
    nntile::kernel::norm_fiber::cuda<T>(
        stream.stream,
        m,
        n,
        k,
        batch,
        alpha,
        dev1.ptr,
        beta,
        dev2.ptr,
        dev_dst.ptr);
    auto gpu_dst = wrap::from_device(dev_dst.ptr, dst.size(), stream.stream);
    {
        ::nnhaul::Handle a(sizeof(T) * src1.size());
        ::nnhaul::Handle b(sizeof(T) * src2.size());
        ::nnhaul::Handle c(sizeof(T) * dst.size());
        wrap::store(a, src1);
        wrap::store(b, src2);
        wrap::store(c, dst);
        nntile::nnhaul::norm_fiber.submit<std::tuple<T>>(
            wrap::cuda_worker, m, n, k, batch, alpha, a, beta, b, c, 0);
        wrap::expect_equal(
            "norm_fiber cuda", gpu_dst, wrap::load<T>(c, dst.size()));
    }
    // Chunked-handle variants (the NATIVE chunked fiber driver:
    // whole-fiber pieces chained through the kernel's piecewise hypot
    // merge - algebraically exact but rounded differently from the
    // single call, so the chunked legs compare with the norm-family
    // tolerance; small fibers stay unchunked and bitwise).
    for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * src1.size()))
    {
        {
            wrap::Chunked a(sizeof(T) * src1.size(), chunk);
            wrap::Chunked b(sizeof(T) * src2.size(), chunk);
            wrap::Chunked c(sizeof(T) * dst.size(), chunk);
            wrap::store(a.handle, src1);
            wrap::store(b.handle, src2);
            wrap::store(c.handle, dst);
            nntile::nnhaul::norm_fiber.submit<std::tuple<T>>(
                wrap::cpu_worker, m, n, k, batch, alpha, a.handle, beta,
                b.handle, c.handle, 0);
            wrap::expect_close(
                "norm_fiber cpu chunked", cpu_dst,
                wrap::load<T>(c.handle, dst.size()), 16);
        }
        {
            wrap::Chunked a(sizeof(T) * src1.size(), chunk);
            wrap::Chunked b(sizeof(T) * src2.size(), chunk);
            wrap::Chunked c(sizeof(T) * dst.size(), chunk);
            wrap::store(a.handle, src1);
            wrap::store(b.handle, src2);
            wrap::store(c.handle, dst);
            nntile::nnhaul::norm_fiber.submit<std::tuple<T>>(
                wrap::cuda_worker, m, n, k, batch, alpha, a.handle, beta,
                b.handle, c.handle, 0);
            wrap::expect_close(
                "norm_fiber cuda chunked", gpu_dst,
                wrap::load<T>(c.handle, dst.size()), 16);
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
