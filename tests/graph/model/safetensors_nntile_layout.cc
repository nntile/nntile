/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/safetensors_nntile_layout.cc
 * Unit tests for SafeTensors C-order → NNTile Fortran layout conversion.
 *
 * @version 1.1.0
 * */

#include "test_safetensors_nntile_layout.hh"

#include <catch2/catch_test_macros.hpp>

using nntile::test::safetensors_nntile_layout::c_safetensors_to_nntile_fortran;

TEST_CASE(
    "c_safetensors_to_nntile_fortran permutes 2-D C-order payload",
    "[model][layout]")
{
    const std::vector<std::int64_t> shape{2, 3};
    const std::vector<float> raw{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    std::vector<float> out;
    c_safetensors_to_nntile_fortran(
        reinterpret_cast<const std::uint8_t *>(raw.data()),
        shape,
        out);
    REQUIRE(out.size() == 6);
    REQUIRE(out[0] == 0.f);
    REQUIRE(out[1] == 3.f);
    REQUIRE(out[2] == 1.f);
    REQUIRE(out[3] == 4.f);
    REQUIRE(out[4] == 2.f);
    REQUIRE(out[5] == 5.f);
}

TEST_CASE(
    "c_safetensors_to_nntile_fortran permutes 3-D C-order payload",
    "[model][layout]")
{
    const std::vector<std::int64_t> shape{2, 2, 2};
    std::vector<float> raw(8);
    for(std::size_t i = 0; i < raw.size(); ++i)
    {
        raw[i] = static_cast<float>(i);
    }
    std::vector<float> out;
    c_safetensors_to_nntile_fortran(
        reinterpret_cast<const std::uint8_t *>(raw.data()),
        shape,
        out);
    REQUIRE(out.size() == 8);
    REQUIRE(out[0] == 0.f);
    REQUIRE(out[1] == 4.f);
    REQUIRE(out[2] == 2.f);
    REQUIRE(out[3] == 6.f);
    REQUIRE(out[4] == 1.f);
    REQUIRE(out[5] == 5.f);
    REQUIRE(out[6] == 3.f);
    REQUIRE(out[7] == 7.f);
}
