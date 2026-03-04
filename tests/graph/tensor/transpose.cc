/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/transpose.cc
 * Test TensorGraph transpose operation against nntile::tensor::transpose.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/transpose.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/transpose.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Scalar alpha = 1.0;
constexpr Index ndim = 1;

} // anonymous namespace

template<typename T>
void check_transpose_vs_tensor_api(
    const std::vector<Index>& shape,
    Scalar alpha,
    Index ndim)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // Output shape: output_shape[i] = src_shape[(i + ndim) % ndim]
    std::vector<Index> dst_shape(shape.size());
    for(size_t i = 0; i < shape.size(); ++i)
    {
        dst_shape[i] = shape[(i + ndim) % shape.size()];
    }
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("transpose_test");
    auto* src_node = graph.data(shape, "src", DataType::FP32);
    src_node->mark_input(true);

    auto* dst_node = gt::transpose(alpha, src_node, "dst", ndim);
    dst_node->mark_output(true);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> src_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i));
    }

    runtime.bind_data("src", src_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path (same input data) ---
    nntile::tensor::TensorTraits src_traits(shape, shape);
    nntile::tensor::TensorTraits dst_traits(dst_shape, dst_shape);
    std::vector<int> distr_src(src_traits.grid.nelems, 0);
    std::vector<int> distr_dst(dst_traits.grid.nelems, 0);
    nntile::tensor::Tensor<T> src(src_traits, distr_src);
    nntile::tensor::Tensor<T> dst(dst_traits, distr_dst);

    {
        auto tile = src.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            loc[i] = static_cast<Y>(src_data[i]);
        }
        loc.release();
    }

    nntile::tensor::transpose<T>(alpha, src, dst, ndim);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(dst_nelems);
    {
        auto tile = dst.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < dst_nelems; ++i)
        {
            tensor_result[i] = static_cast<float>(loc[i]);
        }
        loc.release();
    }

    constexpr float tol = 1e-5f;
    REQUIRE(graph_result.size() == tensor_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        REQUIRE(std::abs(graph_result[i] - tensor_result[i]) < tol);
    }
}

TEST_CASE("TensorGraph transpose structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto* src = graph.data({dim0, dim1}, "src");

    auto* dst = gt::transpose(alpha, src, "dst", ndim);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape()[0] == dim1);
    REQUIRE(dst->shape()[1] == dim0);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "TRANSPOSE");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph transpose rejects duplicate tensors", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;
    TensorGraph graph("test");
    auto* src = graph.data({dim0, dim1}, "src");

    REQUIRE_THROWS_AS(gt::transpose(alpha, src, src, Index(1)), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph transpose matches nntile::tensor::transpose", "[graph][tensor]")
{
    const auto [alpha, ndim, shape] = GENERATE(
        std::tuple{1.0, Index(1), std::vector<Index>{4, 5}},
        std::tuple{2.0, Index(1), std::vector<Index>{4, 5}},
        std::tuple{1.0, Index(1), std::vector<Index>{3, 6}},
        std::tuple{1.0, Index(1), std::vector<Index>{2, 3, 4}},
        std::tuple{1.0, Index(2), std::vector<Index>{2, 3, 4}});

    check_transpose_vs_tensor_api<nntile::fp32_t>(shape, alpha, ndim);
}
