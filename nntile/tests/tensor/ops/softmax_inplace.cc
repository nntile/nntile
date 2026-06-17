/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/softmax_inplace.cc
 * Test TensorGraph softmax_inplace operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/softmax_inplace.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tensor/ops/maxsumexp.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/clear.hh"
#include "nntile/tensor/ops/maxsumexp.hh"
#include "nntile/tensor/ops/softmax_inplace.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr Index axis_0 = 0;
constexpr Index axis_1 = 1;
constexpr int redux = 0;
constexpr Scalar alpha_one = 1.0;
constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

static std::vector<Index> maxsumexp_dst_shape(
    const std::vector<Index> &src_shape, Index axis)
{
    std::vector<Index> dst;
    for (Index i = 0; i < static_cast<Index>(src_shape.size()); ++i)
    {
        if (i != axis)
        {
            dst.push_back(src_shape[i]);
        }
    }
    dst.push_back(2);
    return dst;
}

TEST_CASE("TensorGraph softmax_inplace structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto *maxsumexp_node = graph.data({dim1, dim0, 2})->set_name("maxsumexp");
    auto *dst = graph.data({dim1, dim0})->set_name("dst");

    gt::softmax_inplace(maxsumexp_node, dst, alpha_one, axis_1);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SOFTMAX_INPLACE");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph softmax_inplace rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *mse = graph.data({5, 4, 2})->set_name("mse");
    auto *dst = graph.data({5, 4})->set_name("dst");

    REQUIRE_THROWS_AS(gt::softmax_inplace(nullptr, dst, alpha_one, axis_1),
        std::invalid_argument);
    REQUIRE_THROWS_AS(gt::softmax_inplace(mse, nullptr, alpha_one, axis_1),
        std::invalid_argument);
}