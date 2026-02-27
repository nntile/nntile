/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/autograd_add_example.cc
 * Example: Torch-like autograd with chain of add operations.
 *
 * Forward:  w = x + y,  z = w + u.
 * Backward: each tensor (x, y, u, w) gets its gradient.
 *
 * @version 1.1.0
 * */

#include <iostream>

#include <nntile/graph.hh>

int main()
{
    using namespace nntile::graph;

    NNGraph g("autograd_add_example");

    // Leaves: x, y, u
    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32);
    auto* u = g.tensor({2, 3}, "u", DataType::FP32);

    // Chain: w = x + y,  z = w + u (graph deduced from x, y, u)
    auto* w = add(nntile::Scalar(1.0), x, nntile::Scalar(1.0), y, "w");
    auto* z = add(nntile::Scalar(1.0), w, nntile::Scalar(1.0), u, "z");

    std::cout << "=== Forward chain ===" << std::endl;
    std::cout << "  w = x + y" << std::endl;
    std::cout << "  z = w + u  (= x + y + u)" << std::endl;
    std::cout << "  Leaves: x, y, u (is_leaf=true)" << std::endl;
    std::cout << "  Intermediate: w (has_producer)" << std::endl;
    std::cout << "  Output: z (has_producer)" << std::endl;

    // Set grad_z = 1, then backward
    auto* z_grad = g.get_or_create_grad(z, "z_grad");
    fill(nntile::Scalar(1.0), z_grad->data());
    z->backward();

    std::cout << "\n=== After z.backward() ===" << std::endl;
    std::cout << "  x.has_grad()=" << x->has_grad()
              << ", y.has_grad()=" << y->has_grad()
              << ", u.has_grad()=" << u->has_grad()
              << ", w.has_grad()=" << w->has_grad() << std::endl;
    std::cout << "  Each tensor gets grad: grad_x, grad_y, grad_u, grad_w"
              << std::endl;

    std::cout << "\n=== Graph structure ===" << std::endl;
    std::cout << g.to_string() << std::endl;

    std::cout << "Autograd add chain example completed successfully." << std::endl;
    return 0;
}
