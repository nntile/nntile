/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/sumprod_slice.cc
 * sumprod_slice: direct kernel versus the NNHaul wrapper on the same
 * inputs.
 *
 * @version 1.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/sumprod_slice.hh>
#include <nntile/nnhaul/ops/sumprod_slice.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index m, Index n, Index k, Scalar alpha, Scalar beta,
    bool run_cpu_legs = true)
{
    using Y = typename T::repr_t;
    std::vector<T> src1(static_cast<std::size_t>(m * n * k));
    std::vector<T> src2(src1.size());
    std::vector<T> dst(static_cast<std::size_t>(m * n));
    for (Index i = 0; i < m * n * k; ++i)
    {
        src1[static_cast<std::size_t>(i)] = Y(i + 1) / Y(i * i + 1);
        src2[static_cast<std::size_t>(i)] = Y(i * i + 1);
    }
    for (Index i = 0; i < m * n; ++i)
    {
        dst[static_cast<std::size_t>(i)] = Y(-i - 1);
    }
    auto cpu_dst = dst;
    if (run_cpu_legs)
    {
        std::cout << "Run kernel::sumprod_slice::cpu\n";
        nntile::kernel::sumprod_slice::cpu<T>(
            m, n, k, alpha, src1.data(), src2.data(), beta, cpu_dst.data());
        {
            ::nnhaul::Handle a(sizeof(T) * src1.size());
            ::nnhaul::Handle b(sizeof(T) * src2.size());
            ::nnhaul::Handle c(sizeof(T) * dst.size());
            wrap::store(a, src1);
            wrap::store(b, src2);
            wrap::store(c, dst);
            nntile::nnhaul::sumprod_slice.submit<std::tuple<T>>(
                wrap::cpu_worker, m, n, k, alpha, a, b, beta, c);
            wrap::expect_equal(
                "sumprod_slice cpu", cpu_dst, wrap::load<T>(c, dst.size()));
        }
    }
    wrap::Stream stream;
    wrap::Device<T> dev1(src1);
    wrap::Device<T> dev2(src2);
    wrap::Device<T> dev_dst(dst);
    std::cout << "Run kernel::sumprod_slice::cuda\n";
    nntile::kernel::sumprod_slice::cuda<T>(
        stream.stream, m, n, k, alpha, dev1.ptr, dev2.ptr, beta, dev_dst.ptr);
    auto gpu_dst = wrap::from_device(dev_dst.ptr, dst.size(), stream.stream);
    {
        ::nnhaul::Handle a(sizeof(T) * src1.size());
        ::nnhaul::Handle b(sizeof(T) * src2.size());
        ::nnhaul::Handle c(sizeof(T) * dst.size());
        wrap::store(a, src1);
        wrap::store(b, src2);
        wrap::store(c, dst);
        nntile::nnhaul::sumprod_slice.submit<std::tuple<T>>(
            wrap::cuda_worker, m, n, k, alpha, a, b, beta, c);
        wrap::expect_equal(
            "sumprod_slice cuda", gpu_dst, wrap::load<T>(c, dst.size()));
    }
    // Chunked-handle variants (staged fallback), same references. At
    // big sizes the host and device k-loops round differently (the
    // file's CPU comparison is bitwise), so those legs compare against
    // the device reference only.
    for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * src1.size()))
    {
        if (run_cpu_legs)
        {
            wrap::Chunked a(sizeof(T) * src1.size(), chunk);
            wrap::Chunked b(sizeof(T) * src2.size(), chunk);
            wrap::Chunked c(sizeof(T) * dst.size(), chunk);
            wrap::store(a.handle, src1);
            wrap::store(b.handle, src2);
            wrap::store(c.handle, dst);
            nntile::nnhaul::sumprod_slice.submit<std::tuple<T>>(
                wrap::cpu_worker, m, n, k, alpha, a.handle, b.handle, beta,
                c.handle);
            wrap::expect_equal(
                "sumprod_slice cpu chunked", cpu_dst,
                wrap::load<T>(c.handle, dst.size()));
        }
        {
            wrap::Chunked a(sizeof(T) * src1.size(), chunk);
            wrap::Chunked b(sizeof(T) * src2.size(), chunk);
            wrap::Chunked c(sizeof(T) * dst.size(), chunk);
            wrap::store(a.handle, src1);
            wrap::store(b.handle, src2);
            wrap::store(c.handle, dst);
            nntile::nnhaul::sumprod_slice.submit<std::tuple<T>>(
                wrap::cuda_worker, m, n, k, alpha, a.handle, b.handle, beta,
                c.handle);
            wrap::expect_equal(
                "sumprod_slice cuda chunked", gpu_dst,
                wrap::load<T>(c.handle, dst.size()));
        }
    }
}

int main()
{
    wrap::Session session;
    (void)session;
    check<fp32_t>(3, 5, 7, 2.0, -1.0);
    check<fp32_t>(3, 5, 7, -1.0, 0.0);
    check<fp64_t>(3, 5, 7, 2.0, -1.0);
    check<fp64_t>(3, 5, 7, -1.0, 0.0);
    // Larger leg (40000 bytes: one run under the process class). k==1
    // keeps one product per output so the device kernel's shared-memory
    // atomicAdd order cannot change the bits (larger k diverges by an
    // ulp between runs - the comparison stays bitwise like the rest of
    // the file).
    check<fp32_t>(100, 100, 1, 2.0, -1.0);
    return 0;
}
