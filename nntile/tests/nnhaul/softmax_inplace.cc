/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/softmax_inplace.cc
 * softmax_inplace: direct kernel versus the NNHaul wrapper on the same
 * inputs.
 *
 * @version 2.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/softmax_inplace.hh>
#include <nntile/nnhaul/ops/softmax_inplace.hh>

#include <cmath>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index m, Index n, Index k, bool run_cpu_legs = true)
{
    using Y = typename T::repr_t;
    Scalar const alpha = 1.0;
    std::vector<T> maxsumexp(static_cast<std::size_t>(2 * m * n));
    for (Index i = 0; i < 2 * m * n; i += 2)
    {
        maxsumexp[static_cast<std::size_t>(i)] = Y(i);
        maxsumexp[static_cast<std::size_t>(i + 1)] = Y(std::exp(Y(i) / Y{10}));
    }
    std::vector<T> dst(static_cast<std::size_t>(m * n * k));
    for (Index i = 0; i < m * n * k; ++i)
    {
        dst[static_cast<std::size_t>(i)] = Y(-i - 1);
    }
    auto cpu_dst = dst;
    if (run_cpu_legs)
    {
        std::cout << "Run kernel::softmax_inplace::cpu\n";
        nntile::kernel::softmax_inplace::cpu<T>(
            m, n, k, maxsumexp.data(), alpha, cpu_dst.data());
        {
            ::nnhaul::Handle mx(sizeof(T) * maxsumexp.size());
            ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
            wrap::store(mx, maxsumexp);
            wrap::store(dst_h, dst);
            nntile::nnhaul::softmax_inplace.submit<std::tuple<T>>(
                wrap::cpu_worker, m, n, k, mx, alpha, dst_h);
            wrap::expect_close(
                "softmax_inplace cpu", cpu_dst,
                wrap::load<T>(dst_h, dst.size()), 4);
        }
    }
    wrap::Stream stream;
    wrap::Device<T> dev_mx(maxsumexp);
    wrap::Device<T> dev_dst(dst);
    std::cout << "Run kernel::softmax_inplace::cuda\n";
    nntile::kernel::softmax_inplace::cuda<T>(
        stream.stream, m, n, k, dev_mx.ptr, alpha, dev_dst.ptr);
    auto gpu_dst = wrap::from_device(dev_dst.ptr, dst.size(), stream.stream);
    {
        ::nnhaul::Handle mx(sizeof(T) * maxsumexp.size());
        ::nnhaul::Handle dst_h(sizeof(T) * dst.size());
        wrap::store(mx, maxsumexp);
        wrap::store(dst_h, dst);
        nntile::nnhaul::softmax_inplace.submit<std::tuple<T>>(
            wrap::cuda_worker, m, n, k, mx, alpha, dst_h);
        wrap::expect_equal(
            "softmax_inplace cuda", gpu_dst, wrap::load<T>(dst_h, dst.size()));
    }
    // Chunked-handle variants (the NATIVE chunked unit driver: whole-row
    // pieces of the elementwise map, bitwise-equal to the plain call),
    // same references and tolerances. At big sizes the host reference
    // hits subnormal outputs (host exp versus device expf differ by one
    // subnormal ulp), so those legs compare against the device reference
    // only.
    for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * dst.size()))
    {
        if (run_cpu_legs)
        {
            wrap::Chunked mx(sizeof(T) * maxsumexp.size(), chunk);
            wrap::Chunked dst_h(sizeof(T) * dst.size(), chunk);
            wrap::store(mx.handle, maxsumexp);
            wrap::store(dst_h.handle, dst);
            nntile::nnhaul::softmax_inplace.submit<std::tuple<T>>(
                wrap::cpu_worker, m, n, k, mx.handle, alpha, dst_h.handle);
            wrap::expect_close(
                "softmax_inplace cpu chunked", cpu_dst,
                wrap::load<T>(dst_h.handle, dst.size()), 4);
        }
        {
            wrap::Chunked mx(sizeof(T) * maxsumexp.size(), chunk);
            wrap::Chunked dst_h(sizeof(T) * dst.size(), chunk);
            wrap::store(mx.handle, maxsumexp);
            wrap::store(dst_h.handle, dst);
            nntile::nnhaul::softmax_inplace.submit<std::tuple<T>>(
                wrap::cuda_worker, m, n, k, mx.handle, alpha, dst_h.handle);
            wrap::expect_equal(
                "softmax_inplace cuda chunked", gpu_dst,
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
    // Big fp32 legs (dst = 100*100*20*4 = 800000 bytes: both chunk
    // variants chunk) through the staged fallback, device reference.
    check<fp32_t>(100, 100, 20, false);
    return 0;
}
