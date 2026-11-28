/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file nntile/tests/nnhaul/norm.cc
 * norm: direct kernel versus the NNHaul wrapper on the same inputs.
 *
 * @version 2.0.0
 * */

#include "harness.hh"

#include <tuple>

#include <nntile/kernel/norm.hh>
#include <nntile/nnhaul/ops/norm.hh>

using nntile::Index;
using nntile::Scalar;
using nntile::fp32_t;
using nntile::fp64_t;

template<typename T>
void check(Index nelems, Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    std::vector<T> src(static_cast<std::size_t>(nelems), T(Y{1.0}));
    std::vector<T> dst(1, T(Y{0.0}));
    auto cpu_dst = dst;
    std::cout << "Run kernel::norm::cpu\n";
    nntile::kernel::norm::cpu<T>(nelems, alpha, src.data(), beta, cpu_dst.data());
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T));
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::norm.submit<std::tuple<T>>(
            wrap::cpu_worker, nelems, alpha, src_h, beta, dst_h, 0);
        wrap::expect_close("norm cpu", cpu_dst, wrap::load<T>(dst_h, 1), 4);
    }
    wrap::Stream stream;
    wrap::Device<T> dev_src(src);
    wrap::Device<T> dev_dst(dst);
    std::cout << "Run kernel::norm::cuda\n";
    nntile::kernel::norm::cuda<T>(
        stream.stream, nelems, alpha, dev_src.ptr, beta, dev_dst.ptr);
    auto gpu_dst = wrap::from_device(dev_dst.ptr, 1, stream.stream);
    {
        ::nnhaul::Handle src_h(sizeof(T) * src.size());
        ::nnhaul::Handle dst_h(sizeof(T));
        wrap::store(src_h, src);
        wrap::store(dst_h, dst);
        nntile::nnhaul::norm.submit<std::tuple<T>>(
            wrap::cuda_worker, nelems, alpha, src_h, beta, dst_h, 0);
        wrap::expect_equal("norm cuda", gpu_dst, wrap::load<T>(dst_h, 1));
    }
    // Chunked-handle variants (the NATIVE chunked driver: per-chunk
    // pieces chained through the kernel's own hypot beta contract -
    // equal to the single call up to the rounding of the piecewise
    // hypot chain, so the chunked legs compare with the family
    // tolerance instead of bitwise; scalar dst stays unchunked).
    for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * src.size()))
    {
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            ::nnhaul::Handle dst_h(sizeof(T));
            wrap::store(src_h.handle, src);
            wrap::store(dst_h, dst);
            nntile::nnhaul::norm.submit<std::tuple<T>>(
                wrap::cpu_worker, nelems, alpha, src_h.handle, beta, dst_h, 0);
            wrap::expect_close(
                "norm cpu chunked", cpu_dst, wrap::load<T>(dst_h, 1), 4);
        }
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            ::nnhaul::Handle dst_h(sizeof(T));
            wrap::store(src_h.handle, src);
            wrap::store(dst_h, dst);
            nntile::nnhaul::norm.submit<std::tuple<T>>(
                wrap::cuda_worker, nelems, alpha, src_h.handle, beta, dst_h,
                0);
            wrap::expect_close(
                "norm cuda chunked", gpu_dst, wrap::load<T>(dst_h, 1), 4);
        }
    }
}

template<typename T>
void check_zero(Index nelems, Index zero_prefix)
{
    // Zero-region legs (the family-1 zero-anchor class for the
    // whole-tile op): a leading all-zero run covers whole chunk pieces,
    // so the FIRST pieces of the hypot chain contribute norm 0 and the
    // chain must still define dst exactly like the plain kernel does.
    using Y = typename T::repr_t;
    std::vector<T> src(static_cast<std::size_t>(nelems), T(Y{0.0}));
    for (Index i = zero_prefix; i < nelems; ++i)
    {
        src[static_cast<std::size_t>(i)] = Y(1 + (i - zero_prefix) % 13);
    }
    for (Scalar beta : {Scalar{0.0}, Scalar{0.5}, Scalar{1.0}})
    {
        std::vector<T> dst(1, T(Y(-2.5)));
        auto cpu_dst = dst;
        nntile::kernel::norm::cpu<T>(nelems, Scalar{2.0}, src.data(), beta,
            cpu_dst.data());
        for (std::size_t chunk : wrap::chunk_variants(sizeof(T) * src.size()))
        {
            wrap::Chunked src_h(sizeof(T) * src.size(), chunk);
            ::nnhaul::Handle dst_h(sizeof(T));
            wrap::store(src_h.handle, src);
            wrap::store(dst_h, dst);
            nntile::nnhaul::norm.submit<std::tuple<T>>(
                wrap::cuda_worker, nelems, Scalar{2.0}, src_h.handle, beta,
                dst_h, 0);
            // The fp64 leg is two class chunks at the 1 MiB class, so
            // the hypot chain spans a chunk boundary; the fp32 leg is
            // one run. Both must match the single-call tolerance.
            wrap::expect_close("norm zero-region chunked", cpu_dst,
                wrap::load<T>(dst_h, 1), 16);
        }
        // The all-zero tile: every piece contributes 0; beta must still
        // land (fabs(beta*dst) / 0), the family-1 zero-anchor contract.
        std::vector<T> zsrc(static_cast<std::size_t>(nelems), T(Y{0.0}));
        std::vector<T> zdst(1, T(Y(-2.5)));
        auto zcpu = zdst;
        nntile::kernel::norm::cpu<T>(nelems, Scalar{2.0}, zsrc.data(), beta,
            zcpu.data());
        wrap::Chunked zsrc_h(sizeof(T) * zsrc.size(), std::size_t{12288});
        ::nnhaul::Handle zdst_h(sizeof(T));
        wrap::store(zsrc_h.handle, zsrc);
        wrap::store(zdst_h, zdst);
        nntile::nnhaul::norm.submit<std::tuple<T>>(
            wrap::cuda_worker, nelems, Scalar{2.0}, zsrc_h.handle, beta,
            zdst_h, 0);
        wrap::expect_equal("norm all-zero chunked", zcpu,
            wrap::load<T>(zdst_h, 1));
    }
}

int main()
{
    wrap::Session session;
    (void)session;
    check<fp32_t>(1, 1.0, 0.0);
    check<fp32_t>(10000, 1.0, 0.0);
    check<fp64_t>(1, 1.0, 0.0);
    check<fp64_t>(10000, 1.0, 0.0);
    check<fp32_t>(200000, 1.0, 0.0);
    // Zero-region legs: leading zeros cover whole pieces, so the
    // pieces contribute only their absolute value to the chain.
    check_zero<fp32_t>(200000, 10000);
    check_zero<fp64_t>(200000, 5000);
    return 0;
}
