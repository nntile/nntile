/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile/chunked_units_math.cc
 * Address-arithmetic tests for the shared chunk walker: the power-of-two
 * shift/mask fast path (the production chunk class) must be bit-for-bit
 * the exact division/multiply fallback (the test-only
 * NNHAUL_CHUNK_UNITS_FORCE_BYTES regime), for both the flat chunk map
 * and the UnitSpan row/panel walker.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/nnhaul/chunk_units.hh"

#include <cstddef>
#include <vector>

namespace
{

namespace cu = nntile::haul::chunkunits;

//! Build an unchunked Buffer view over host memory (ChunkView's
//! non-`real` path only addresses the one block).
template<typename T>
::nnhaul::Buffer make_buffer(std::vector<T> &data)
{
    ::nnhaul::Buffer b;
    b.ptr_ = data.data();
    b.capacity_ = data.size() * sizeof(T);
    b.nbytes_ = data.size() * sizeof(T);
    b.chunk_count_ = 1;
    b.chunk_bytes_ = 0;
    return b;
}

//! Verify every element maps exactly as integer divmod would, and that
//! the public chunk_of/chunk_off/chunk_begin/chunk_end agree.
template<typename T>
void check_exact(cu::ChunkView<T> const &cv, T const *base)
{
    REQUIRE(cv.chunk_elems > 0);
    std::size_t const expected_n =
        (cv.total_elems + cv.chunk_elems - 1) / cv.chunk_elems;
    REQUIRE(cv.nchunks == expected_n);
    for(std::size_t e = 0; e < cv.total_elems; ++e)
    {
        std::size_t const c = e / cv.chunk_elems;
        std::size_t const off = e - c * cv.chunk_elems;
        REQUIRE(cv.chunk_of(e) == c);
        REQUIRE(cv.chunk_off(e) == off);
        REQUIRE(cv.chunk_begin(c) == c * cv.chunk_elems);
        REQUIRE(cv.chunk_begin(c) <= e);
        REQUIRE(e < cv.chunk_end(c));
        REQUIRE(cv.ptr(c, off) == base + e);
    }
    for(std::size_t c = 0; c < cv.nchunks; ++c)
    {
        std::size_t const end =
            (c + 1) * cv.chunk_elems < cv.total_elems
            ? (c + 1) * cv.chunk_elems
            : cv.total_elems;
        REQUIRE(cv.chunk_end(c) == end);
        REQUIRE(cv.ptr(c, 0) == base + c * cv.chunk_elems);
    }
}

template<typename T>
void check_unit_span(cu::ChunkView<T> const &cv, T const *base,
    std::size_t unit)
{
    cu::UnitSpan<T> span{&cv, unit, 1};
    for(std::size_t r = 0; r < cv.total_elems / unit; ++r)
    {
        std::size_t const s = r * unit;
        bool const straddles = !cu::in_one_chunk(cv, s, unit);
        REQUIRE(span.straddles(static_cast<nntile::Index>(r))
            == straddles);
        if(!straddles)
        {
            REQUIRE(span.ptr(static_cast<nntile::Index>(r)) == base + s);
        }
        nntile::Index const run = span.run(static_cast<nntile::Index>(r));
        REQUIRE(run >= 0);
        if(!straddles)
        {
            REQUIRE(run >= 1);
            REQUIRE(static_cast<std::size_t>(run) <=
                (cv.total_elems - s) / unit);
        }
    }
}

} // namespace

TEST_CASE("chunk walker: power-of-two shift/mask equals divmod",
    "[chunked][units]")
{
    for(std::size_t ce : {std::size_t{1}, std::size_t{2}, std::size_t{8},
            std::size_t{256}, std::size_t{1024}})
    {
        std::vector<float> data(4096);
        auto buf = make_buffer(data);
        cu::ChunkView<float> cv(&buf, ce * sizeof(float));
        REQUIRE(cv.pow2);
        REQUIRE(cv.shift == cu::log2_floor(ce));
        REQUIRE(cv.mask == ce - 1);
        check_exact(cv, data.data());
        check_unit_span(cv, data.data(), ce < 4 ? 1 : 4);
    }
}

TEST_CASE("chunk walker: non-power-of-two fallback is exact",
    "[chunked][units]")
{
    // 300 elements per virtual chunk: neither a power of two nor a
    // divisor of the 4096-element space.
    std::vector<double> data(4096);
    auto buf = make_buffer(data);
    cu::ChunkView<double> cv(&buf, 300 * sizeof(double));
    REQUIRE(!cv.pow2);
    REQUIRE(cv.chunk_elems == 300);
    check_exact(cv, data.data());
    check_unit_span(cv, data.data(), 7);
}

TEST_CASE("chunk walker: unchunked buffer stays the 1-chunk case",
    "[chunked][units]")
{
    std::vector<float> pow2_data(1024);
    auto pow2_buf = make_buffer(pow2_data);
    cu::ChunkView<float> pow2_view(&pow2_buf, 0);
    REQUIRE(pow2_view.nchunks == 1);
    REQUIRE(pow2_view.pow2);
    check_exact(pow2_view, pow2_data.data());

    std::vector<float> odd_data(1000);
    auto odd_buf = make_buffer(odd_data);
    cu::ChunkView<float> odd_view(&odd_buf, 0);
    REQUIRE(odd_view.nchunks == 1);
    REQUIRE(!odd_view.pow2);
    check_exact(odd_view, odd_data.data());
}
