// /*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
//  *                              (Skoltech), Russia. All rights reserved.
//  *                 2023-present Artificial Intelligence Research Institute
//  *                              (AIRI), Russia. All rights reserved.
//  *
//  * NNTile is software framework for fast training of big neural networks on
//  * distributed-memory heterogeneous systems based on StarPU runtime system.
//  *
//  * @file tests/graph/compiled/gather.cc
//  * Test for compiled graph gather operation.
//  *
//  * @version 1.1.0
//  * */

// #include "compiled_test_utils.hh"

// #include "nntile/tensor/gather.hh"

// using namespace nntile;
// using namespace nntile::graph;
// using namespace nntile::graph::test;

// TEST_CASE_METHOD(
//     GraphTestFixture,
//     "CompiledGraph Gather vs Tensor",
//     "[graph][verification]")
// {
//     auto build_graph = [](LogicalGraph& g) {
//         auto& x = g.tensor({4, 6}, "x", DataType::FP32);
//         auto& index = g.tensor({2, 3}, "index", DataType::INT64);
//         auto& y = gather(x, index, "y", 0);
//     };

//     auto run_tensor_direct = [](std::map<std::string, std::vector<float>>& inputs,
//                                std::map<std::string, std::vector<float>>& outputs,
//                                const nntile::Context&) {
//         using T = nntile::fp32_t;
//         nntile::tensor::TensorTraits x_traits({4, 6}, {4, 6});
//         nntile::tensor::Tensor<T> x(x_traits);
//         nntile::tensor::TensorTraits index_traits({2, 3}, {2, 3});
//         nntile::tensor::Tensor<nntile::int64_t> index_tensor(index_traits);
//         nntile::tensor::TensorTraits y_traits({2, 3, 6}, {2, 3, 6});
//         nntile::tensor::Tensor<T> y(y_traits);

//         std::vector<long long> index_data = {0, 1, 2, 3, 0, 1};
//         write_tensor(x, inputs["x"]);
//         write_tensor(index_tensor, index_data);
//         nntile::tensor::gather<T>(x, index_tensor, y, 0);
//         outputs["y"] = read_tensor(y);
//     };

//     verify_graph_vs_tensor<nntile::fp32_t>(
//         build_graph, run_tensor_direct,
//         {"x"}, {"y"}, context
//     );
// }