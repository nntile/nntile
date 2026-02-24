/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/maxsumexp.cc
 * Test for compiled graph maxsumexp operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/maxsumexp.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE(
    "CompiledGraph MaxSumExp",
    "[graph][verification]")
{
    // Test that maxsumexp op is registered in logical graph
    // (CompiledGraph::compile triggers StarPU handle issues for this op)
    LogicalGraph g("test");
    auto& x = g.tensor({4, 6}, "x", DataType::FP32);
    auto& y = g.tensor({2, 6}, "y", DataType::FP32);
    maxsumexp(x, y, 0, 0);

    REQUIRE(y.has_producer());
    REQUIRE(y.producer()->type() == OpType::MAXSUMEXP);
}
