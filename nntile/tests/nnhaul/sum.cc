/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/sum.cc
 * sum: direct kernel versus the NNHaul wrapper on the same inputs.
 *
 * @version 1.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/sum.hh>
#include <nntile/nnhaul/ops/sum.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index nelems, Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    auto src = wrap::series<T>(static_cast<std::size_t>(nelems), 1, 1);
    std::vector<T> dst(1, T(Y(-1)));
    auto cpu_dst = dst;
    std::cout << "Run kernel::sum::cpu\n";
    nntile::kernel::sum::cpu<T>(nelems, alpha, src.data(), beta, cpu_dst.data());
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T));
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::sum.submit<std::tuple<T>>(
            wrap::cpu_worker, nelems, alpha, src_h, beta, dst_h);
        wrap::expect_equal("sum cpu", cpu_dst, wrap::load<T>(dst_h, 1));
    }
    wrap::Stream stream;
    wrap::Device<T> dev_src(src);
    wrap::Device<T> dev_dst(dst);
    std::cout << "Run kernel::sum::cuda\n";
    nntile::kernel::sum::cuda<T>(
        stream.stream, nelems, alpha, dev_src.ptr, beta, dev_dst.ptr);
    auto gpu_dst = wrap::from_device(dev_dst.ptr, 1, stream.stream);
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T));
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::sum.submit<std::tuple<T>>(
            wrap::cuda_worker, nelems, alpha, src_h, beta, dst_h);
        wrap::expect_equal("sum cuda", gpu_dst, wrap::load<T>(dst_h, 1));
    }
    // Chunked-handle variants (the NATIVE chunked driver: per-chunk
    // pieces chained through the kernel's own Kahan beta contract -
    // algebraically exact but rounded differently from the single call,
    // so the chunked legs compare with the family tolerance instead of
    // bitwise; the scalar dst stays unchunked - mixed-chunking
    // coverage).
    for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * src.size()))
    {
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            ::nnhaul::Handle dst_h(sizeof(T));
            wrap::store(src_h.handle, src);
            wrap::store(dst_h, dst);
            nntile::nnhaul::sum.submit<std::tuple<T>>(
                wrap::cpu_worker, nelems, alpha, src_h.handle, beta, dst_h);
            wrap::expect_close(
                "sum cpu chunked", cpu_dst, wrap::load<T>(dst_h, 1), 16);
        }
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            ::nnhaul::Handle dst_h(sizeof(T));
            wrap::store(src_h.handle, src);
            wrap::store(dst_h, dst);
            nntile::nnhaul::sum.submit<std::tuple<T>>(
                wrap::cuda_worker, nelems, alpha, src_h.handle, beta, dst_h);
            wrap::expect_close(
                "sum cuda chunked", gpu_dst, wrap::load<T>(dst_h, 1), 16);
        }
    }
}

int main()
{
    wrap::Session session;
    (void)session;
    check<fp32_t>(35, 1.0, -1.0);
    check<fp32_t>(35, 2.0, 0.0);
    check<fp32_t>(35, 0.0, 1.0);
    check<fp64_t>(35, 1.0, -1.0);
    check<fp64_t>(35, 2.0, 0.0);
    check<fp64_t>(35, 0.0, 1.0);
    check<fp32_t>(200000, 1.0, -1.0);
    return 0;
}
