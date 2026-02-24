/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/randn.cc
 * Test for compiled graph randn operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/randn.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE(
    "CompiledGraph Randn",
    "[graph][verification]")
{
    // For random operations, we just check that the operation can be added to a graph
    nntile::graph::LogicalGraph g("test");
    auto& x = g.tensor({4, 6}, "x", nntile::DataType::FP32);
    nntile::graph::randn(x, {0, 0}, {4, 6}, 0, 0.0f, 1.0f);
    // Just check that the operation was added to the graph
    REQUIRE(x.has_producer());
    REQUIRE(x.producer()->type() == nntile::graph::OpType::RANDN);
}
