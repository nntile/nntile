#include <nntile/tensor/tensor_ref.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/gemm.cc
 * Test TileGraph gemm vs nntile::core (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

#include "context_fixture.hh"
#include "gemm_core_reference.hh"
#include "gemm_test_shapes.hh"
#include "nntile/constants.hh"
#include "nntile/core/gemm.hh"
#include "nntile/core/tile.hh"
#include "nntile/tile.hh"
#include "nntile/tile/ops/gemm.hh"
#include "test_frobenius.hh"

using namespace nntile;
namespace tg = nntile::tile;

namespace
{

std::vector<float> run_tile_graph_gemm(const std::vector<Index> &a_shape,
    const std::vector<Index> &b_shape,
    const std::vector<Index> &c_shape,
    const std::vector<float> &a_data,
    const std::vector<float> &b_data,
    Scalar alpha,
    bool trans_a_flag,
    bool trans_b_flag,
    Index ndim_flag,
    Index batch_ndim_flag)
{
    TileGraph graph("tile_gemm");
    auto *a = graph.data(a_shape, "a", DataType::FP32);
    auto *b = graph.data(b_shape, "b", DataType::FP32);
    auto *c = graph.data(c_shape, "c", DataType::FP32);

    tg::gemm(a,
        b,
        c,
        alpha,
        Scalar(0),
        trans_a_flag,
        trans_b_flag,
        ndim_flag,
        batch_ndim_flag);

    Runtime runtime(graph);
    runtime.compile();

    const Index c_nelems = std::accumulate(c_shape.begin(),
        c_shape.end(),
        Index{1},
        std::multiplies<Index>{});
    std::vector<float> c_init(static_cast<size_t>(c_nelems), 0.f);

    runtime.bind_data(a, a_data);
    runtime.bind_data(b, b_data);
    runtime.bind_data(c, c_init);
    runtime.execute();
    runtime.wait();
    return runtime.get_output<float>(c);
}

} // namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TileGraph gemm matches core",
    "[graph][tile]")
{
    const auto [trans_a_flag, trans_b_flag, ndim_flag, batch_ndim_flag, alpha] =
        GENERATE(
            std::tuple{false, false, Index(1), Index(0), Scalar(1.0)},
            std::tuple{false, true, Index(1), Index(0), Scalar(1.0)},
            std::tuple{true, false, Index(1), Index(0), Scalar(1.0)},
            std::tuple{true, true, Index(1), Index(0), Scalar(1.0)},
            std::tuple{false, false, Index(2), Index(0), Scalar(0.5)},
            std::tuple{false, true, Index(2), Index(0), Scalar(0.5)},
            std::tuple{true, false, Index(2), Index(0), Scalar(0.5)},
            std::tuple{true, true, Index(2), Index(0), Scalar(0.5)},
            std::tuple{false, false, Index(1), Index(1), Scalar(1.0)},
            std::tuple{false, true, Index(1), Index(1), Scalar(1.0)},
            std::tuple{true, false, Index(1), Index(1), Scalar(1.0)},
            std::tuple{true, true, Index(1), Index(1), Scalar(1.0)});

    const auto [a_shape, b_shape] = nntile::test::gemm_test_shapes(
        trans_a_flag, trans_b_flag, ndim_flag, batch_ndim_flag);
    const std::vector<Index> c_shape = nntile::tensor::gemm_output_shape(
        a_shape, b_shape, trans_a_flag, trans_b_flag, ndim_flag, batch_ndim_flag);

    const Index a_nelems = std::accumulate(a_shape.begin(),
        a_shape.end(),
        Index{1},
        std::multiplies<Index>{});
    const Index b_nelems = std::accumulate(b_shape.begin(),
        b_shape.end(),
        Index{1},
        std::multiplies<Index>{});

    using Y = nntile::fp32_t::repr_t;
    std::vector<float> a_data(static_cast<size_t>(a_nelems));
    std::vector<float> b_data(static_cast<size_t>(b_nelems));
    for (Index i = 0; i < a_nelems; ++i)
    {
        a_data[static_cast<size_t>(i)] =
            static_cast<float>(Y(i % 10)) * 0.1f;
    }
    for (Index i = 0; i < b_nelems; ++i)
    {
        b_data[static_cast<size_t>(i)] =
            static_cast<float>(Y(i % 7)) * 0.15f;
    }

    const std::vector<float> core_out = nntile::test::core_gemm_reference_fp32(
        a_shape,
        b_shape,
        a_data,
        b_data,
        alpha,
        trans_a_flag,
        trans_b_flag,
        ndim_flag,
        batch_ndim_flag);
    const std::vector<float> tile_out = run_tile_graph_gemm(a_shape,
        b_shape,
        c_shape,
        a_data,
        b_data,
        alpha,
        trans_a_flag,
        trans_b_flag,
        ndim_flag,
        batch_ndim_flag);

    nntile::test::require_relative_frobenius_error(tile_out, core_out);
}
