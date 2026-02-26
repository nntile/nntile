/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/fill.cc
 * Test for compiled graph fill operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/fill.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

namespace
{

template<typename T>
DataType dtype_for();

template<>
DataType dtype_for<fp32_t>()
{
    return DataType::FP32;
}

template<>
DataType dtype_for<fp64_t>()
{
    return DataType::FP64;
}

template<typename T>
void check_fill(const std::vector<Index>& shape, Scalar val,
                const nntile::Context& context)
{
    auto build_graph = [shape, val](LogicalGraph& g) {
        auto& x = g.tensor(shape, "x", dtype_for<T>());
        fill(val, x);
    };

    auto run_tensor_direct = [shape, val](
        std::map<std::string, std::vector<typename T::repr_t>>&,
        std::map<std::string, std::vector<typename T::repr_t>>& outputs,
        const nntile::Context&)
    {
        nntile::tensor::TensorTraits x_traits(shape, shape);
        nntile::tensor::Tensor<T> x(x_traits);

        nntile::tensor::fill<T>(val, x);
        outputs["x"] = read_tensor(x);
    };

    verify_graph_vs_tensor<T>(
        build_graph, run_tensor_direct,
        {}, {"x"}, context
    );
}

} // namespace

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Fill vs Tensor",
    "[graph][verification]")
{
    const Scalar val = -0.5;
    const std::vector<std::vector<Index>> shapes = {
        {},
        {5},
        {11},
        {11, 12, 13}
    };

    for(const auto& shape : shapes)
    {
        check_fill<nntile::fp32_t>(shape, val, context);
    }

    for(const auto& shape : shapes)
    {
        check_fill<nntile::fp64_t>(shape, val, context);
    }
}
