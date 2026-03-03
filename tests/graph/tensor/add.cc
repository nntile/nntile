/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/add.cc
 * Test TensorGraph add operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "nntile/graph/tensor/add.hh"
#include "nntile/graph/tensor.hh"

using namespace nntile::graph;

TEST_CASE("TensorGraph add operation", "[graph]")
{
    TensorGraph graph("test");

    // Create input data nodes
    auto* x = graph.data({4, 5}, "x");
    auto* y = graph.data({4, 5}, "y");

    // Add operation: z = 1.0*x + 1.0*y
    auto* z = add(1.0, x, 1.0, y, "z");

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(z->shape()[0] == 4);
    REQUIRE(z->shape()[1] == 5);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "ADD");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == z);
}
