/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/linear_manual_example.cc
 * Example: module with custom backward (GradMode + wrap_with_module_op).
 *
 * PyTorch-like pattern: when a module has custom backward, forward runs in
 * no_grad so autograd ops (gemm, add_fiber) don't build a graph; then the
 * output is wrapped in a single ModuleOp whose backward is the module's
 * build_backward. Thus output.backward() invokes the custom backward.
 *
 * @version 1.1.0
 * */

#include <iostream>

#include <nntile.hh>

int main()
{
    nntile::Context context(1, 0, 0, "/tmp/nntile_ooc", 16777216, 0,
                            "localhost", 5001, 0);

    nntile::graph::NNGraph graph("LinearManual_Graph");

    // LinearManual: custom backward overrides autograd of gemm/add_fiber
    nntile::module::LinearManual linear(
        graph, "linear_manual", 8, 4, true,
        nntile::graph::DataType::FP32);

    auto* input = graph.tensor(
        {4, 8}, "input", nntile::graph::DataType::FP32, true);

    // build_forward runs gemm/add_fiber in GradMode::Guard (no producer),
    // then wrap_with_module_op attaches our custom backward
    auto& output = linear.build_forward(*input);

    // Set grad and run backward - invokes LinearManual::build_backward
    graph.get_or_create_grad(&output, "output_grad");
    nntile::graph::fill(nntile::Scalar(1.0), output.grad()->data());
    output.backward();

    std::cout << "LinearManual (custom backward) example:\n"
              << "  output.has_producer() = " << output.has_producer() << "\n"
              << "  weight.grad = " << (linear.weight_tensor()->has_grad() ? "yes" : "no") << "\n"
              << "  bias.grad = " << (linear.bias_tensor()->has_grad() ? "yes" : "no") << "\n"
              << "  input.grad = " << (input->has_grad() ? "yes" : "no") << "\n";

    return 0;
}
