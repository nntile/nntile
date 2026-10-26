/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile/chunked_swap_conv_randn_ab.cc
 * A/B of the chunk-addressed swap_two_axes, conv2d family and non-dense
 * randn WRAPPER drivers: each op runs through the ordinary nntile wrapper
 * (submit/body) on an unchunked and on a multi-chunk (compile-time
 * ::nnhaul::kChunkClassBytes class, tail included) handle, and the two
 * outputs must be bitwise identical. Part of the outputs is also compared
 * against a host reference for the pure-copy swap.
 *
 * The shapes exceed the chunk class, so the chunked branch is exercised
 * end to end; conv cases cover several strides/dilations/offsets and
 * beta != 0, randn covers dense and foreign-strided geometries.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "context_fixture.hh"

#include <nntile/defs.h>

#ifdef NNTILE_USE_CUDA
#ifdef NNTILE_USE_NNHAUL

#include <nnhaul/nnhaul.hh>

#include <nntile/base_types.hh>
#include <nntile/nnhaul/ops/conv2d_bwd_input_inplace.hh>
#include <nntile/nnhaul/ops/conv2d_bwd_weight_inplace.hh>
#include <nntile/nnhaul/ops/conv2d_inplace.hh>
#include <nntile/nnhaul/ops/randn.hh>
#include <nntile/nnhaul/ops/swap_two_axes.hh>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace nntile;
namespace nh = ::nnhaul;

namespace
{

std::size_t bytes_of(std::size_t nelems)
{
    return nelems * sizeof(fp32_t);
}

template<typename T>
void store(nh::Handle &h, std::vector<T> const &src)
{
    auto held = h.acquire(nh::Access::Write);
    std::memcpy(held.get(), src.data(), src.size() * sizeof(T));
}

template<typename T>
std::vector<T> load(nh::Handle &h, std::size_t count)
{
    nh::wait();
    auto held = h.acquire(nh::Access::Read);
    std::vector<T> out(count);
    std::memcpy(out.data(), held.get(), count * sizeof(T));
    return out;
}

//! Run one body on an unchunked and on a chunked handle set and require
//! bitwise equality of the produced output buffer(s).
template<typename Run>
std::vector<fp32_t> ab(std::vector<std::size_t> const &nbytes,
    std::vector<std::vector<fp32_t>> const &inputs,
    std::vector<std::size_t> const &output_ids, std::vector<fp32_t> const &zero,
    Run &&run)
{
    std::vector<std::vector<fp32_t>> legs;
    for (int chunked = 0; chunked < 2; ++chunked)
    {
        std::vector<std::unique_ptr<nh::Handle>> handles;
        handles.reserve(nbytes.size());
        for (std::size_t b : nbytes)
        {
            handles.emplace_back(chunked
                ? std::make_unique<nh::Handle>(b, nh::kChunkClassBytes)
                : std::make_unique<nh::Handle>(b));
        }
        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            store(*handles[i], inputs[i]);
        }
        for (std::size_t o : output_ids)
        {
            store(*handles[o], zero);
        }
        run(handles);
        legs.push_back(load<fp32_t>(*handles[output_ids[0]], zero.size()));
        nh::wait();
    }
    REQUIRE(legs[0].size() == legs[1].size());
    REQUIRE(std::memcmp(legs[0].data(), legs[1].data(),
        legs[0].size() * sizeof(fp32_t)) == 0);
    return legs[1];
}

// ---------------------------------------------------------------------------
// swap_two_axes
// ---------------------------------------------------------------------------

std::vector<fp32_t> swap_ref(std::size_t d0, std::size_t d1,
    std::size_t d2, std::size_t d3, std::size_t d4,
    std::vector<fp32_t> const &src)
{
    std::vector<fp32_t> dst(src.size());
    for (std::size_t i0 = 0; i0 < d0; ++i0)
    {
        for (std::size_t i1 = 0; i1 < d1; ++i1)
        {
            for (std::size_t i2 = 0; i2 < d2; ++i2)
            {
                for (std::size_t i3 = 0; i3 < d3; ++i3)
                {
                    for (std::size_t i4 = 0; i4 < d4; ++i4)
                    {
                        std::size_t const s =
                            ((((i0 * d1 + i1) * d2 + i2) * d3 + i3) * d4)
                            + i4;
                        std::size_t const t =
                            ((((i0 * d3 + i3) * d2 + i2) * d1 + i1) * d4)
                            + i4;
                        dst[t] = src[s];
                    }
                }
            }
        }
    }
    return dst;
}

void check_swap(Index d0, Index d1, Index d2, Index d3, Index d4)
{
    std::size_t const nelems = static_cast<std::size_t>(d0) * d1 * d2 * d3 * d4;
    std::vector<fp32_t> src(nelems);
    for (std::size_t i = 0; i < nelems; ++i)
    {
        src[i] = fp32_t(0.25F * static_cast<float>(i % 577) - 3.0F);
    }
    std::vector<fp32_t> const ref = swap_ref(d0, d1, d2, d3, d4, src);
    std::vector<fp32_t> const out = ab({bytes_of(nelems), bytes_of(nelems)},
        {src}, {1}, std::vector<fp32_t>(nelems, 0.0F),
        [&](std::vector<std::unique_ptr<nh::Handle>> &h)
        {
            haul::swap_two_axes.submit<std::tuple<fp32_t>>(0, d0, d1, d2, d3,
                d4, *h[0], *h[1]);
        });
    REQUIRE(std::memcmp(out.data(), ref.data(), bytes_of(nelems)) == 0);
}

// ---------------------------------------------------------------------------
// conv2d family
// ---------------------------------------------------------------------------

void check_conv2d_fwd(Index src1_m, Index src1_n, Index src1_channels,
    Index batch, Index src2_m, Index src2_n, Index dilation_m,
    Index dilation_n, Index dst_channels, Index offset_m, Index offset_n,
    Scalar alpha, Index dst_m, Index dst_n, Index stride_m, Index stride_n,
    Scalar beta)
{
    std::size_t const n1 = static_cast<std::size_t>(src1_m) * src1_n
        * src1_channels * batch;
    std::size_t const n2 = static_cast<std::size_t>(src2_m) * src2_n
        * src1_channels * dst_channels;
    std::size_t const nd = static_cast<std::size_t>(dst_m) * dst_n
        * dst_channels * batch;
    std::vector<fp32_t> src1(n1);
    std::vector<fp32_t> src2(n2);
    for (std::size_t i = 0; i < n1; ++i)
    {
        src1[i] = fp32_t(0.01F * static_cast<float>(i % 97) - 0.4F);
    }
    for (std::size_t i = 0; i < n2; ++i)
    {
        src2[i] = fp32_t(0.02F * static_cast<float>(i % 53) - 0.5F);
    }
    std::vector<fp32_t> dst0(nd);
    for (std::size_t i = 0; i < nd; ++i)
    {
        dst0[i] = fp32_t(beta == 0.0 ? 0.0F
            : 0.03F * static_cast<float>(i % 31) - 0.2F);
    }
    ab({bytes_of(n1), bytes_of(n2), bytes_of(nd)}, {src1, src2}, {2},
        dst0, [&](std::vector<std::unique_ptr<nh::Handle>> &h)
        {
            haul::conv2d_inplace.submit<std::tuple<fp32_t>>(0, src1_m, src1_n,
                src1_channels, batch, src2_m, src2_n, dilation_m,
                dilation_n, dst_channels, offset_m, offset_n, alpha, *h[0],
                *h[1], dst_m, dst_n, stride_m, stride_n, beta, *h[2]);
        });
}

void check_conv2d_bwd_input(Index src1_m, Index src1_n, Index stride_m,
    Index stride_n, Index src1_channels, Index batch, Index src2_m,
    Index src2_n, Index dilation_m, Index dilation_n, Index dst_channels,
    Index offset_m, Index offset_n, Scalar alpha, Index dst_m, Index dst_n,
    Scalar beta)
{
    std::size_t const n1 = static_cast<std::size_t>(src1_m) * src1_n
        * src1_channels * batch;
    std::size_t const n2 = static_cast<std::size_t>(src2_m) * src2_n
        * dst_channels * src1_channels;
    std::size_t const nd = static_cast<std::size_t>(dst_m) * dst_n
        * dst_channels * batch;
    std::vector<fp32_t> src1(n1);
    std::vector<fp32_t> src2(n2);
    for (std::size_t i = 0; i < n1; ++i)
    {
        src1[i] = fp32_t(0.01F * static_cast<float>(i % 89) - 0.3F);
    }
    for (std::size_t i = 0; i < n2; ++i)
    {
        src2[i] = fp32_t(0.02F * static_cast<float>(i % 47) - 0.4F);
    }
    std::vector<fp32_t> dst0(nd);
    for (std::size_t i = 0; i < nd; ++i)
    {
        dst0[i] = fp32_t(beta == 0.0 ? 0.0F
            : 0.03F * static_cast<float>(i % 29) - 0.1F);
    }
    ab({bytes_of(n1), bytes_of(n2), bytes_of(nd)}, {src1, src2}, {2},
        dst0, [&](std::vector<std::unique_ptr<nh::Handle>> &h)
        {
            haul::conv2d_bwd_input_inplace.submit<std::tuple<fp32_t>>(0, src1_m,
                src1_n, stride_m, stride_n, src1_channels, batch, src2_m,
                src2_n, dilation_m, dilation_n, dst_channels, offset_m,
                offset_n, alpha, *h[0], *h[1], dst_m, dst_n, beta, *h[2]);
        });
}

void check_conv2d_bwd_weight(Index src1_m, Index src1_n,
    Index src1_channels, Index batch, Index src2_m, Index src2_n,
    Index stride_m, Index stride_n, Index src2_channels, Index offset_m,
    Index offset_n, Scalar alpha, Index dst_m, Index dst_n,
    Index dilation_m, Index dilation_n, Scalar beta)
{
    std::size_t const n1 = static_cast<std::size_t>(src1_m) * src1_n
        * src1_channels * batch;
    std::size_t const n2 = static_cast<std::size_t>(src2_m) * src2_n
        * src2_channels * batch;
    std::size_t const nd = static_cast<std::size_t>(dst_m) * dst_n
        * src1_channels * src2_channels;
    std::vector<fp32_t> src1(n1);
    std::vector<fp32_t> src2(n2);
    for (std::size_t i = 0; i < n1; ++i)
    {
        src1[i] = fp32_t(0.01F * static_cast<float>(i % 83) - 0.3F);
    }
    for (std::size_t i = 0; i < n2; ++i)
    {
        src2[i] = fp32_t(0.02F * static_cast<float>(i % 43) - 0.4F);
    }
    std::vector<fp32_t> dst0(nd);
    for (std::size_t i = 0; i < nd; ++i)
    {
        dst0[i] = fp32_t(beta == 0.0 ? 0.0F
            : 0.03F * static_cast<float>(i % 23) - 0.1F);
    }
    ab({bytes_of(n1), bytes_of(n2), bytes_of(nd)}, {src1, src2}, {2},
        dst0, [&](std::vector<std::unique_ptr<nh::Handle>> &h)
        {
            haul::conv2d_bwd_weight_inplace.submit<std::tuple<fp32_t>>(0,
                src1_m, src1_n, src1_channels, batch, src2_m, src2_n,
                stride_m, stride_n, src2_channels, offset_m, offset_n, alpha,
                *h[0], *h[1], dst_m, dst_n, dilation_m, dilation_n, beta,
                *h[2]);
        });
}

// ---------------------------------------------------------------------------
// randn
// ---------------------------------------------------------------------------

void check_randn(std::size_t nbytes, Index ndim,
    unsigned long long seed, std::vector<Index> const &start,
    std::vector<Index> const &shape, std::vector<Index> const &stride,
    std::vector<Index> const &underlying_shape)
{
    std::vector<fp32_t> zero(nbytes / sizeof(fp32_t));
    ab({nbytes}, {}, {0}, zero,
        [&](std::vector<std::unique_ptr<nh::Handle>> &h)
        {
            Index nelems = 1;
            for (Index s : shape)
            {
                nelems *= s;
            }
            haul::randn.submit<std::tuple<fp32_t>>(0, ndim, nelems, seed,
                0.0F, 1.0F, start, shape, stride, underlying_shape, *h[0]);
        });
}

} // namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "chunked wrapper A/B: swap_two_axes", "[chunked][swap]")
{
    // Small / degenerate / d4 == 1 shapes (unchunked legs are still
    // compared with the chunked leg, so the chunked kernel's map is
    // checked even when only the reference leg is unchunked).
    check_swap(2, 3, 4, 5, 6);
    check_swap(1, 17, 1, 13, 1);
    check_swap(2, 1, 7, 1, 1);
    check_swap(3, 5, 2, 7, 1);
    // Multi-chunk, d4 > 1 and d4 == 1 (both > the 16 MiB class).
    check_swap(4, 64, 32, 48, 32);
    check_swap(2, 1200, 3, 1400, 1);
    nh::wait();
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "chunked wrapper A/B: conv2d_inplace", "[chunked][conv2d]")
{
    // Dense stride, no padding: src1 17.3 MiB (2 chunks incl. tail),
    // dst 34 MiB (3 chunks).
    check_conv2d_fwd(520, 520, 16, 1, 3, 3, 1, 1, 32, 0, 0, 1.0F,
        518, 518, 1, 1, 0.0F);
    // Stride 2, dilation 2, positive padding, beta != 0.
    check_conv2d_fwd(520, 520, 16, 1, 3, 3, 2, 2, 32, 1, 1, 0.75F,
        259, 259, 2, 2, 0.5F);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "chunked wrapper A/B: conv2d_bwd_input_inplace", "[chunked][conv2d]")
{
    check_conv2d_bwd_input(262, 262, 2, 2, 64, 1, 3, 3, 2, 2, 32, 1, 1,
        1.0F, 524, 524, 0.0F);
    check_conv2d_bwd_input(262, 262, 1, 1, 64, 1, 3, 3, 1, 1, 32, 0, 0,
        0.5F, 264, 264, 0.25F);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "chunked wrapper A/B: conv2d_bwd_weight_inplace", "[chunked][conv2d]")
{
    // src1 17.3 MiB and src2 34 MiB: both multi-chunk.
    check_conv2d_bwd_weight(520, 520, 16, 1, 518, 518, 1, 1, 32, 0, 0,
        1.0F, 3, 3, 1, 1, 0.0F);
    // Nonzero padding and beta.
    check_conv2d_bwd_weight(518, 518, 16, 1, 260, 260, 2, 2, 32, 1, 1,
        0.5F, 3, 3, 1, 1, 0.5F);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "chunked wrapper A/B: randn dense and non-dense", "[chunked][randn]")
{
    // Dense Fortran geometry, 32 MiB -> 3 chunks.
    check_randn(bytes_of(8u << 20), 2, 12345ULL, {0, 0},
        {1u << 20, 8}, {1, 1u << 20}, {1u << 20, 8});
    // Foreign strided tile: stride[1] != shape[0], 16 MiB of values and
    // a 33.5 MiB flat footprint (3 chunks) and a nonzero start.
    check_randn(bytes_of(8388608u), 2, 987654321ULL, {1, 2},
        {4096, 1024}, {1, 8192}, {4096, 2048});
}

#endif // NNTILE_USE_NNHAUL
#endif // NNTILE_USE_CUDA
