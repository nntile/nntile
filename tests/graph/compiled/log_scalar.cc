/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/log_scalar.cc
 * Test for compiled graph log_scalar operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/log_scalar.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE(
    "CompiledGraph LogScalar",
    "[graph][verification]")
{
    // Test that log_scalar operation compiles
    nntile::graph::LogicalGraph g("test");
    auto& x = g.tensor({1}, "x", nntile::DataType::FP32);
    nntile::graph::log_scalar(x, "test_value");

    // Just check that the operation was added to the graph
    REQUIRE(x.has_producer());
    REQUIRE(x.producer()->type() == nntile::graph::OpType::LOG_SCALAR);
}