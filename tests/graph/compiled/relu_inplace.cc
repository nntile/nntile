#include "compiled_test_utils.hh"

#include "nntile/tensor/relu_inplace.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph ReluInplace vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& x = g.tensor({4, 6}, "x", DataType::FP32);
        relu_inplace(x);
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits x_traits({4, 6}, {4, 6});
        nntile::tensor::Tensor<T> x(x_traits);

        write_tensor(x, inputs["x"]);
        nntile::tensor::relu_inplace<T>(x);
        outputs["x"] = read_tensor(x);
    };

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"x"}, {"x"}, context
    );
}
