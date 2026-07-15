/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/conv2d_inplace.cc
 * Test TensorGraph conv2d_inplace operation against
 * nntile::tensor::conv2d_inplace.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/conv2d_inplace.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/conv2d_inplace.hh"
#include "nntile/tensor.hh"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

// Y shape from X shape (W,H,C_in,N), C shape (K_x,K_y,C_in,C_out), padding,
// stride, dilation
std::vector<Index> conv2d_output_shape(const std::vector<Index> &x_shape,
    const std::vector<Index> &c_shape,
    const std::array<Index, 2> &padding,
    const std::array<Index, 2> &stride,
    const std::array<Index, 2> &dilation)
{
    Index W_out =
        (x_shape[0] + 2 * padding[0] - dilation[0] * (c_shape[0] - 1) - 1) /
            stride[0] +
        1;
    Index H_out =
        (x_shape[1] + 2 * padding[1] - dilation[1] * (c_shape[1] - 1) - 1) /
            stride[1] +
        1;
    return {W_out, H_out, c_shape[3], x_shape[3]};
}

} 

TEST_CASE("TensorGraph conv2d_inplace structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    nntile::TensorRef x = graph.data({4, 4, 2, 2});
    x->set_name("x");
    nntile::TensorRef c = graph.data({2, 2, 2, 2});
    c->set_name("c");
    nntile::TensorRef y = graph.data({3, 3, 2, 2});
    y->set_name("y");

    gt::conv2d_inplace(1.0, x, c, 0.0, y, {0, 0}, {1, 1}, {1, 1});

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "CONV2D_INPLACE");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == y);
}

TEST_CASE("TensorGraph conv2d_inplace rejects null tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef x = graph.data({4, 4, 2, 2});
    x->set_name("x");
    nntile::TensorRef c = graph.data({2, 2, 2, 2});
    c->set_name("c");
    nntile::TensorRef y = graph.data({3, 3, 2, 2});
    y->set_name("y");

    REQUIRE_THROWS_AS(
        gt::conv2d_inplace(1.0, nullptr, c, 0.0, y, {0, 0}, {1, 1}, {1, 1}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::conv2d_inplace(1.0, x, nullptr, 0.0, y, {0, 0}, {1, 1}, {1, 1}),
        std::invalid_argument);
}

