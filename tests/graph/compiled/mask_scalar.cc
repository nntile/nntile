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

    std::vector<nntile::bool_t> mask_data(24);
    for(size_t i = 0; i < mask_data.size(); ++i)
    {
        mask_data[i] = nntile::bool_t((i % 2) == 0);
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

    write_tensor(mask_tensor, mask_data);
    write_tensor(x_tensor, x_data);

    nntile::tensor::mask_scalar<nntile::fp32_t>(mask_tensor, 0.5f, x_tensor, 0);
    auto tensor_out = read_tensor(x_tensor);

    REQUIRE(graph_out.size() == tensor_out.size());
    for(size_t i = 0; i < graph_out.size(); ++i)
    {
        REQUIRE(graph_out[i] == Catch::Approx(tensor_out[i]).epsilon(1e-5));
    }
}
