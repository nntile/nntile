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
 * Usage: autograd_add_example [x_requires_grad] [y_requires_grad]
 *   Each arg: 0 or 1 (default 0 if omitted).
 *
 * @version 1.1.0
 * */

#include <iostream>
#include <string>

#include <nntile/graph.hh>

int main(int argc, char** argv)
{
    using namespace nntile::graph;

    if(argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
    {
        std::cout << "Usage: autograd_add_example [x_requires_grad] [y_requires_grad]\n"
                     "  Diamond graph: w=x+y, v=w+y, z=v+w. Run z.backward().\n"
                     "  Each arg: 0 or 1 (default 0 if omitted).\n";
        return 0;
    }

    bool x_requires_grad = (argc > 1 && std::string(argv[1]) != "0");
    bool y_requires_grad = (argc > 2 && std::string(argv[2]) != "0");

    NNGraph g("autograd_add_example");

    // Leaves: x, y (requires_grad from argv)
    auto* x = g.tensor({2, 3}, "x", DataType::FP32, x_requires_grad);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32, y_requires_grad);

    // Diamond: w = x + y,  v = w + y,  z = v + w
    auto* w = add(nntile::Scalar(1.0), x, nntile::Scalar(1.0), y, "w");
    auto* v = add(nntile::Scalar(1.0), w, nntile::Scalar(1.0), y, "v");
    auto* z = add(nntile::Scalar(1.0), v, nntile::Scalar(1.0), w, "z");

    std::cout << "=== Forward (diamond) ===" << std::endl;
    std::cout << "  w = x + y,  v = w + y,  z = v + w" << std::endl;
    std::cout << "  x_requires_grad=" << x_requires_grad
              << ", y_requires_grad=" << y_requires_grad << std::endl;

    // Set grad_z = 1, then backward
    auto* z_grad = g.get_or_create_grad(z, "z_grad");
    fill(nntile::Scalar(1.0), z_grad->data());
    z->backward();

    std::cout << "\n=== After z.backward() ===" << std::endl;
    std::cout << "  x.has_grad()=" << x->has_grad()
              << ", y.has_grad()=" << y->has_grad() << std::endl;

    size_t add_inplace_count = 0;
    for(const auto& op : g.logical_graph().ops())
    {
        if(op->type() == OpType::ADD_INPLACE)
        {
            ++add_inplace_count;
        }
    }
    std::cout << "  ADD_INPLACE count=" << add_inplace_count << std::endl;

    std::cout << "\n=== Graph structure ===" << std::endl;
    std::cout << g.to_string() << std::endl;

    std::cout << "Autograd add diamond example completed successfully."
              << std::endl;
    return 0;
}
