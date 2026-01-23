#include "compiled_test_utils.hh"

#include "nntile/tensor/relu.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Relu vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& x = g.tensor({4, 6}, "x", DataType::FP32);
        relu(x, "y");
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits x_traits({4, 6}, {4, 6});
        nntile::tensor::Tensor<T> x(x_traits);
        nntile::tensor::TensorTraits y_traits({4, 6}, {4, 6});
        nntile::tensor::Tensor<T> y(y_traits);

        write_tensor(x, inputs["x"]);
        nntile::tensor::relu<T>(x, y);
        outputs["y"] = read_tensor(y);
    };

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"x"}, {"y"}, context
    );
}
