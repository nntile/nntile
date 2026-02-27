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
 * With x and y having requires_grad=false, backward() adds no ADD_INPLACE ops.
 *
 * @version 1.1.0
 * */

#include <iostream>

#include <nntile/graph.hh>

int main()
{
    using namespace nntile::graph;

    NNGraph g("autograd_add_example");

    // Leaves: x, y (no gradients needed)
    auto* x = g.tensor({2, 3}, "x", DataType::FP32, false);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32, false);

    // Diamond: w = x + y,  v = w + y,  z = v + w
    auto* w = add(nntile::Scalar(1.0), x, nntile::Scalar(1.0), y, "w");
    auto* v = add(nntile::Scalar(1.0), w, nntile::Scalar(1.0), y, "v");
    auto* z = add(nntile::Scalar(1.0), v, nntile::Scalar(1.0), w, "z");

    std::cout << "=== Forward (diamond) ===" << std::endl;
    std::cout << "  w = x + y,  v = w + y,  z = v + w" << std::endl;
    std::cout << "  Leaves: x, y (requires_grad=false)" << std::endl;

    // Set grad_z = 1, then backward
    auto* z_grad = g.get_or_create_grad(z, "z_grad");
    fill(nntile::Scalar(1.0), z_grad->data());
    z->backward();

    std::cout << "\n=== After z.backward() ===" << std::endl;
    std::cout << "  x.has_grad()=" << x->has_grad()
              << ", y.has_grad()=" << y->has_grad()
              << " (no grads: requires_grad was false)" << std::endl;

    // Verify no ADD_INPLACE ops were added (no inputs required grad)
    size_t add_inplace_count = 0;
    for(const auto& op : g.logical_graph().ops())
    {
        if(op->type() == OpType::ADD_INPLACE)
        {
            ++add_inplace_count;
        }
    }
    std::cout << "  ADD_INPLACE count=" << add_inplace_count
              << " (expected 0: no gradient accumulation)" << std::endl;

    std::cout << "\n=== Graph structure ===" << std::endl;
    std::cout << g.to_string() << std::endl;

    std::cout << "Autograd add diamond example completed successfully."
              << std::endl;
    return 0;
}
