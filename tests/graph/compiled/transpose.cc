#include "compiled_test_utils.hh"

#include "nntile/tensor/transpose.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

template<typename T>
void check_transpose(Index ndim, const std::vector<Index>& input_shape,
                     const nntile::Context& context)
{
    using ValueT = typename T::repr_t;

    // Calculate expected output shape
    std::vector<Index> output_shape(input_shape.size());
    for(size_t i = 0; i < input_shape.size(); ++i)
    {
        output_shape[i] = input_shape[(i + ndim) % input_shape.size()];
    }

    auto build_graph = [&](LogicalGraph& g) {
        auto& x = g.tensor(input_shape, "x", DataType::FP32);
        transpose(x, "y", -1.0f, ndim);
    };

    auto run_tensor_direct = [&](std::map<std::string, std::vector<ValueT>>& inputs,
                                 std::map<std::string, std::vector<ValueT>>& outputs,
                                 const nntile::Context&) {
        nntile::tensor::TensorTraits x_traits(input_shape, input_shape);
        nntile::tensor::Tensor<T> x(x_traits);
        nntile::tensor::TensorTraits y_traits(output_shape, output_shape);
        nntile::tensor::Tensor<T> y(y_traits);

        write_tensor(x, inputs["x"]);
        nntile::tensor::transpose<T>(-1.0f, x, y, ndim);
        outputs["y"] = read_tensor(y);
    };

    verify_graph_vs_tensor<T>(
        build_graph, run_tensor_direct,
        {"x"}, {"y"}, context
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Transpose vs Tensor - 2D ndim=1",
    "[graph][verification]")
{
    check_transpose<fp32_t>(1, {4, 6}, context);
    check_transpose<fp64_t>(1, {4, 6}, context);
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Transpose vs Tensor - 3D ndim=1",
    "[graph][verification]")
{
    check_transpose<fp32_t>(1, {2, 3, 4}, context);
    check_transpose<fp64_t>(1, {2, 3, 4}, context);
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Transpose vs Tensor - 3D ndim=2",
    "[graph][verification]")
{
    check_transpose<fp32_t>(2, {2, 3, 4}, context);
    check_transpose<fp64_t>(2, {2, 3, 4}, context);
}
