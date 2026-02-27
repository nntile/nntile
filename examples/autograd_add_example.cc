/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/autograd_add_example.cc
 * Example: Torch-like autograd with diamond-shaped add operations.
 *
 * Forward:  w = x + y,  v = w + y,  z = v + w (diamond: w feeds into v and z).
 * Backward: each tensor (x, y, w, v) gets its gradient; w.grad accumulates
 * from both v and z.
 *
 * @version 1.1.0
 * */

#include <iostream>

#include <nntile/graph.hh>

int main()
{
    using namespace nntile::graph;

    NNGraph g("autograd_add_example");

    // Leaves: x, y
    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32);

    // Diamond: w = x + y,  v = w + y,  z = v + w
    // w feeds into both v and z; backward must process v and z before w
    auto* w = add(nntile::Scalar(1.0), x, nntile::Scalar(1.0), y, "w");
    auto* v = add(nntile::Scalar(1.0), w, nntile::Scalar(1.0), y, "v");
    auto* z = add(nntile::Scalar(1.0), v, nntile::Scalar(1.0), w, "z");

    std::cout << "=== Forward (diamond) ===" << std::endl;
    std::cout << "  w = x + y" << std::endl;
    std::cout << "  v = w + y" << std::endl;
    std::cout << "  z = v + w  (w feeds into both v and z)" << std::endl;
    std::cout << "  Leaves: x, y (is_leaf=true)" << std::endl;
    std::cout << "  Intermediate: w, v (has_producer)" << std::endl;
    std::cout << "  Output: z (has_producer)" << std::endl;

    // Set grad_z = 1, then backward
    auto* z_grad = g.get_or_create_grad(z, "z_grad");
    fill(nntile::Scalar(1.0), z_grad->data());
    z->backward();

    std::cout << "\n=== After z.backward() ===" << std::endl;
    std::cout << "  x.has_grad()=" << x->has_grad()
              << ", y.has_grad()=" << y->has_grad()
              << ", w.has_grad()=" << w->has_grad()
              << ", v.has_grad()=" << v->has_grad() << std::endl;
    std::cout << "  Each tensor gets grad; w.grad accumulates from v and z"
              << std::endl;

    std::cout << "\n=== Graph structure ===" << std::endl;
    std::cout << g.to_string() << std::endl;

    std::cout << "Autograd add diamond example completed successfully."
              << std::endl;
    return 0;
}
