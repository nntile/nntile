/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/mask_scalar.cc
 * Test for compiled graph mask_scalar operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/mask_scalar.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph MaskScalar vs Tensor",
    "[graph][verification]")
{
    LogicalGraph g("test");
    auto& mask = g.tensor({4, 6}, "mask", DataType::BOOL);
    auto& x = g.tensor({4, 6}, "x", DataType::FP32);
    mask_scalar(mask, x, 0.5f, 0);

    auto compiled = CompiledGraph::compile(g);

    std::vector<char> mask_data(24);
    for(size_t i = 0; i < mask_data.size(); ++i)
    {
        mask_data[i] = static_cast<char>(static_cast<bool>((i % 2) == 0));
    }
    std::vector<float> x_data = make_pattern<float>(24, 0.1f);

    compiled.bind_data("mask", mask_data);
    compiled.bind_data("x", x_data);

    compiled.execute();
    compiled.wait();

    auto graph_out = compiled.get_output<float>("x");

    nntile::tensor::TensorTraits mask_traits({4, 6}, {4, 6});
    nntile::tensor::Tensor<nntile::bool_t> mask_tensor(mask_traits);
    nntile::tensor::TensorTraits x_traits({4, 6}, {4, 6});
    nntile::tensor::Tensor<nntile::fp32_t> x_tensor(x_traits);

    // Use char instead of bool, as std::vector<bool> does not have data member
    std::vector<char> mask_data_repr(24);
    for(size_t i = 0; i < mask_data.size(); ++i)
    {
        mask_data_repr[i] = static_cast<char>(static_cast<bool>(mask_data[i]));
    }
    write_tensor(mask_tensor, mask_data_repr);
    write_tensor(x_tensor, x_data);

    nntile::tensor::mask_scalar<nntile::fp32_t>(mask_tensor, 0.5f, x_tensor, 0);
    auto tensor_out = read_tensor(x_tensor);

    REQUIRE(graph_out.size() == tensor_out.size());
    for(size_t i = 0; i < graph_out.size(); ++i)
    {
        REQUIRE(graph_out[i] == Catch::Approx(tensor_out[i]).epsilon(1e-5));
    }
}
