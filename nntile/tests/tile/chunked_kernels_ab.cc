/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile/chunked_kernels_ab.cc
 * A/B for the chunk-addressed structured kernels: swap_two_axes and
 * transpose are computed through the nntile chunk kernels on both an
 * unchunked and a multi-chunk handle, and compared bitwise against a
 * CPU reference of the stock nntile map. Covers d4 contiguous,
 * d4 == 1, rectangular transposes and tail chunks.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "context_fixture.hh"

#include <nntile/defs.h>

#ifdef NNTILE_USE_CUDA

#include <nntile/kernel/nnhaul/chunk_ops.hh>

#include <nnhaul/nnhaul.hh>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace
{

constexpr std::size_t kMiB = 1024 * 1024;

namespace kops = nntile::kernel::nnhaul;

struct SwapArgs
{
    std::size_t d0;
    std::size_t d1;
    std::size_t d2;
    std::size_t d3;
    std::size_t d4;
    ::nnhaul::ElemType type;
};

struct TransposeArgs
{
    std::size_t m;
    std::size_t n;
    float alpha;
    ::nnhaul::ElemType type;
};

void swap_body(void **buffers, void *cl_args) noexcept
{
    auto *args = static_cast<SwapArgs *>(cl_args);
    auto *src = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto *dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    kops::chunk_swap_two_axes(*src, *dst, args->d0, args->d1, args->d2,
        args->d3, args->d4, args->type);
}

void transpose_body(void **buffers, void *cl_args) noexcept
{
    auto *args = static_cast<TransposeArgs *>(cl_args);
    auto *src = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto *dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    kops::chunk_transpose(*src, *dst, args->m, args->n, args->alpha,
        args->type);
}

template <typename T>
void store(::nnhaul::Handle &h, std::vector<T> const &src)
{
    auto held = h.acquire(::nnhaul::Access::Write);
    std::memcpy(held.get(), src.data(), src.size() * sizeof(T));
}

template <typename T>
std::vector<T> load(::nnhaul::Handle &h, std::size_t count)
{
    ::nnhaul::wait();
    auto held = h.acquire(::nnhaul::Access::Read);
    std::vector<T> out(count);
    std::memcpy(out.data(), held.get(), count * sizeof(T));
    return out;
}

template <typename T>
::nnhaul::ElemType elem_type()
{
    return sizeof(T) == 8 ? ::nnhaul::ElemType::F64
                          : ::nnhaul::ElemType::F32;
}

using T = float;

std::vector<T> swap_ref(std::size_t d0, std::size_t d1, std::size_t d2,
    std::size_t d3, std::size_t d4, std::vector<T> const &src)
{
    std::vector<T> dst(src.size());
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

std::vector<T> transpose_ref(std::size_t m, std::size_t n, float alpha,
    std::vector<T> const &src)
{
    std::vector<T> dst(src.size());
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            dst[i * n + j] = T(alpha * src[i + j * m]);
        }
    }
    return dst;
}

void run_swap(std::size_t d0, std::size_t d1, std::size_t d2,
    std::size_t d3, std::size_t d4)
{
    std::size_t const nelems = d0 * d1 * d2 * d3 * d4;
    std::size_t const nbytes = nelems * sizeof(T);
    std::vector<T> src(nelems);
    for (std::size_t i = 0; i < nelems; ++i)
    {
        src[i] = static_cast<T>(i % 577) * 0.25F - 3.0F;
    }
    std::vector<T> const ref = swap_ref(d0, d1, d2, d3, d4, src);
    std::vector<T> const zero(nelems, 0.0F);

    SwapArgs const args{d0, d1, d2, d3, d4, elem_type<T>()};
    ::nnhaul::Codelet swap("swap-ab", nullptr, swap_body, nullptr);

    std::vector<T> legs[2];
    for (int leg = 0; leg < 2; ++leg)
    {
        std::unique_ptr<::nnhaul::Handle> hs;
        std::unique_ptr<::nnhaul::Handle> hd;
        if (leg == 0)
        {
            hs = std::make_unique<::nnhaul::Handle>(nbytes);
            hd = std::make_unique<::nnhaul::Handle>(nbytes);
        }
        else
        {
            hs = std::make_unique<::nnhaul::Handle>(nbytes,
                ::nnhaul::kChunkClassBytes);
            hd = std::make_unique<::nnhaul::Handle>(nbytes,
                ::nnhaul::kChunkClassBytes);
        }
        store(*hs, src);
        store(*hd, zero);
        ::nnhaul::insert(swap,
            {{&*hs, ::nnhaul::Access::Read},
                {&*hd, ::nnhaul::Access::Write}},
            &args, sizeof(args), 0);
        legs[leg] = load<T>(*hd, nelems);
    }
    if (std::memcmp(legs[0].data(), ref.data(), nbytes) != 0
        || std::memcmp(legs[1].data(), ref.data(), nbytes) != 0)
    {
        std::fprintf(stderr,
            "swap MISMATCH d=(%zu,%zu,%zu,%zu,%zu) chunks=%zu\n", d0, d1,
            d2, d3, d4, (nbytes + ::nnhaul::kChunkClassBytes - 1)
                / ::nnhaul::kChunkClassBytes);
    }
    REQUIRE(std::memcmp(legs[0].data(), ref.data(), nbytes) == 0);
    REQUIRE(std::memcmp(legs[1].data(), ref.data(), nbytes) == 0);
    REQUIRE(std::memcmp(legs[0].data(), legs[1].data(), nbytes) == 0);
}

void run_transpose(std::size_t m, std::size_t n)
{
    std::size_t const nelems = m * n;
    std::size_t const nbytes = nelems * sizeof(T);
    float const alpha = 0.5F;
    std::vector<T> src(nelems);
    for (std::size_t i = 0; i < nelems; ++i)
    {
        src[i] = static_cast<T>(i % 313) * 0.5F - 7.0F;
    }
    std::vector<T> const ref = transpose_ref(m, n, alpha, src);
    std::vector<T> const zero(nelems, 0.0F);

    TransposeArgs const args{m, n, alpha, elem_type<T>()};
    ::nnhaul::Codelet tr("transpose-ab", nullptr, transpose_body, nullptr);

    std::vector<T> legs[2];
    for (int leg = 0; leg < 2; ++leg)
    {
        std::unique_ptr<::nnhaul::Handle> hs;
        std::unique_ptr<::nnhaul::Handle> hd;
        if (leg == 0)
        {
            hs = std::make_unique<::nnhaul::Handle>(nbytes);
            hd = std::make_unique<::nnhaul::Handle>(nbytes);
        }
        else
        {
            hs = std::make_unique<::nnhaul::Handle>(nbytes,
                ::nnhaul::kChunkClassBytes);
            hd = std::make_unique<::nnhaul::Handle>(nbytes,
                ::nnhaul::kChunkClassBytes);
        }
        store(*hs, src);
        store(*hd, zero);
        ::nnhaul::insert(tr,
            {{&*hs, ::nnhaul::Access::Read},
                {&*hd, ::nnhaul::Access::Write}},
            &args, sizeof(args), 0);
        legs[leg] = load<T>(*hd, nelems);
    }
    if (std::memcmp(legs[0].data(), ref.data(), nbytes) != 0
        || std::memcmp(legs[1].data(), ref.data(), nbytes) != 0)
    {
        std::fprintf(stderr, "transpose MISMATCH m=%zu n=%zu chunks=%zu\n",
            m, n, (nbytes + ::nnhaul::kChunkClassBytes - 1)
                / ::nnhaul::kChunkClassBytes);
    }
    REQUIRE(std::memcmp(legs[0].data(), ref.data(), nbytes) == 0);
    REQUIRE(std::memcmp(legs[1].data(), ref.data(), nbytes) == 0);
    REQUIRE(std::memcmp(legs[0].data(), legs[1].data(), nbytes) == 0);
}

} // namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "chunked kernel A/B: swap_two_axes and transpose", "[chunked][kernels]")
{
    // Small / degenerate / d4 == 1 shapes.
    run_swap(2, 3, 4, 5, 6);
    run_swap(1, 17, 1, 13, 1);
    run_swap(2, 1, 7, 1, 1);
    run_swap(3, 5, 2, 7, 1);
    // Multi-chunk (same element count, > 4 MiB): 4*64*32*48*32.
    run_swap(4, 64, 32, 48, 32);
    // Multi-chunk with d4 == 1: 2*600*3*700*1 (~10 MiB).
    run_swap(2, 600, 3, 700, 1);

    // Rectangular transposes, including tail chunks and a straddle.
    run_transpose(3, 5);
    run_transpose(520, 320);
    run_transpose(1024, 1024); // exactly one class: unchunked
    run_transpose(2048, 1024); // two chunks
    run_transpose(1234, 2345); // ~11 MiB, non-multiple tail
    run_transpose(3001, 1001);

    ::nnhaul::wait();
}

#endif // NNTILE_USE_CUDA
