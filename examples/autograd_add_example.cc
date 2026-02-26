/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/autograd_add_example.cc
 * Example: Torch-like autograd for z = add(alpha, x, beta, y) with z.backward().
 *
 * Demonstrates out-of-place add operation and backward gradient propagation
 * mimicking PyTorch's autograd. The TensorNode stores grad_fn (producer op)
 * and backward() builds the gradient graph.
 *
 * @version 1.1.0
 * */

#include <iostream>

#include <nntile/graph.hh>

int main()
{
    using namespace nntile::graph;

    // Create graph
    NNGraph g("autograd_add_example");

    // Create input tensors (leaves)
    auto& x = g.tensor({2, 3}, "x", DataType::FP32);
    auto& y = g.tensor({2, 3}, "y", DataType::FP32);

    // z = alpha * x + beta * y (out-of-place)
    nntile::Scalar alpha = 2.0;
    nntile::Scalar beta = 3.0;
    auto& z = add(g, alpha, x, beta, y, "z");

    std::cout << "=== Forward: z = add(alpha, x, beta, y) ===" << std::endl;
    std::cout << "  alpha=" << alpha << ", beta=" << beta << std::endl;
    std::cout << "  z.grad_fn() = " << (z.grad_fn() ? "ADD" : "null")
              << " (producer op)" << std::endl;
    std::cout << "  x.is_leaf()=" << x.is_leaf()
              << ", y.is_leaf()=" << y.is_leaf() << std::endl;

    // PyTorch-style backward: builds gradient graph
    z.backward();

    std::cout << "\n=== After z.backward() ===" << std::endl;
    std::cout << "  x.has_grad()=" << x.has_grad()
              << ", y.has_grad()=" << y.has_grad() << std::endl;
    std::cout << "  grad_x = alpha * grad_z, grad_y = beta * grad_z"
              << " (with grad_z=1)" << std::endl;

    std::cout << "\n=== Graph structure ===" << std::endl;
    std::cout << g.to_string() << std::endl;

    std::cout << "Autograd add example completed successfully." << std::endl;
    return 0;
}
