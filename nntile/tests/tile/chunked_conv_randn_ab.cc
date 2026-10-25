/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile/chunked_conv_randn_ab.cc
 * A/B of the chunk-addressed conv2d family and randn WRAPPER drivers:
 * each op runs through the ordinary nntile wrapper (submit/body) on an
 * unchunked and on a multi-chunk (4 MiB class, tail included) nnhaul
 * handle, and the two outputs must be bitwise identical.
 *
 * The shapes exceed the chunk class and use both dense and foreign
 * strided output geometries, several strides/dilations/offsets and
 * beta != 0, so the chunked branch is exercised end to end.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "context_fixture.hh"

#include <nnhaul/nnhaul.hh>

#include <nntile/base_types.hh>
#include <nntile/nnhaul/ops/conv2d_bwd_input_inplace.hh>
#include <nntile/nnhaul/ops/conv2d_bwd_weight_inplace.hh>
#include <nntile/nnhaul/ops/conv2d_inplace.hh>
#include <nntile/nnhaul/ops/randn.hh>

#include <cstddef>
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

//! Run one body on an unchunked and on a chunked handle pair and require
//! bitwise equality of every produced buffer. `run` receives the handles.
template<typename Run>
void ab(std::vector<std::size_t> const &nbytes,
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
        legs.push_back(load<fp32_t>(*handles[output_ids[0]],
            zero.size()));
        // Destroy the handles only after the work is drained.
        nh::wait();
    }
    REQUIRE(legs[0].size() == legs[1].size());
    REQUIRE(std::memcmp(legs[0].data(), legs[1].data(),
        legs[0].size() * sizeof(fp32_t)) == 0);
}

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
            haul::conv2d_bwd_weight_inplace.submit<std::tuple<fp32_t>>(0, src1_m,
                src1_n, src1_channels, batch, src2_m, src2_n, stride_m,
                stride_n, src2_channels, offset_m, offset_n, alpha,
                *h[0], *h[1], dst_m, dst_n, dilation_m, dilation_n, beta,
                *h[2]);
        });
}

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
            haul::randn.submit<std::tuple<fp32_t>>(0, ndim, nelems, seed, 0.0F, 1.0F,
                start, shape, stride, underlying_shape, *h[0]);
        });
}

} // namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "chunked wrapper A/B: conv2d_inplace", "[chunked][conv2d]")
{
    // Dense stride, no padding: src1 4.13 MiB (2 chunks incl. tail).
    check_conv2d_fwd(260, 260, 16, 1, 3, 3, 1, 1, 32, 0, 0, 1.0F,
        258, 258, 1, 1, 0.0F);
    // Stride 2, dilation 2, positive padding, beta != 0.
    check_conv2d_fwd(260, 260, 16, 1, 3, 3, 2, 2, 32, 1, 1, 0.75F,
        131, 131, 2, 2, 0.5F);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "chunked wrapper A/B: conv2d_bwd_input_inplace", "[chunked][conv2d]")
{
    check_conv2d_bwd_input(130, 130, 2, 2, 64, 1, 3, 3, 2, 2, 32, 1, 1,
        1.0F, 262, 262, 0.0F);
    check_conv2d_bwd_input(130, 130, 1, 1, 64, 1, 3, 3, 1, 1, 32, 0, 0,
        0.5F, 132, 132, 0.25F);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "chunked wrapper A/B: conv2d_bwd_weight_inplace", "[chunked][conv2d]")
{
    // src1 8.25 MiB and src2 8.13 MiB: both multi-chunk.
    check_conv2d_bwd_weight(260, 260, 16, 1, 258, 258, 1, 1, 32, 0, 0,
        1.0F, 3, 3, 1, 1, 0.0F);
    // Nonzero padding and beta.
    check_conv2d_bwd_weight(258, 258, 16, 1, 130, 130, 2, 2, 32, 1, 1,
        0.5F, 3, 3, 1, 1, 0.5F);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "chunked wrapper A/B: randn dense and non-dense", "[chunked][randn]")
{
    // Dense Fortran geometry, 16 MiB, 4 chunks.
    check_randn(bytes_of(4u << 20), 2, 12345ULL, {0, 0},
        {1u << 20, 4}, {1, 1u << 20}, {1u << 20, 4});
    // Foreign strided tile: stride[1] != shape[0], 8 MiB of values with
    // a 16.8 MiB flat footprint (5 chunks) and a nonzero start.
    check_randn(bytes_of(4190208u), 2, 987654321ULL, {1, 2},
        {4096, 512}, {1, 8192}, {4096, 1024});
}
