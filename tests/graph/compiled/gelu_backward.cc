#include "compiled_test_utils.hh"

#include "nntile/tensor/gelu_backward.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph GeluBackward vs Tensor",
    "[graph][verification]")
{
    auto build_graph = [](LogicalGraph& g) {
        auto& x = g.tensor({3}, "x", DataType::FP32);
        auto& dy = g.tensor({3}, "dy", DataType::FP32);
        auto& dx = g.tensor({3}, "dx", DataType::FP32);
        gelu_backward(x, dy, dx);
    };

    auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
                               std::map<std::string, std::vector<float>>& outputs,
                               const nntile::Context&) {
        using T = nntile::fp32_t;
        nntile::tensor::TensorTraits x_traits({3}, {3});
        nntile::tensor::Tensor<T> x(x_traits);
        nntile::tensor::TensorTraits dy_traits({3}, {3});
        nntile::tensor::Tensor<T> dy(dy_traits);
        nntile::tensor::TensorTraits dx_traits({3}, {3});
        nntile::tensor::Tensor<T> dx(dx_traits);

        write_tensor(x, inputs["x"]);
        write_tensor(dy, inputs["dy"]);
        write_tensor(dx, inputs["dx"]);

        nntile::tensor::gelu_backward<T>(x, dy, dx);
        outputs["dx"] = read_tensor(dx);
    };

    verify_graph_vs_tensor<nntile::fp32_t>(
        build_graph, run_tensor_direct,
        {"x", "dy", "dx"}, {"dx"}, context
    );
}
