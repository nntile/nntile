/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/autograd/autograd.cc
 * Test program for autograd functionality
 *
 * @version 1.1.0
 * */

#include <nntile/autograd/tensor.hh>
#include <nntile/autograd/add.hh>
#include <nntile/autograd/visualize.hh>
#include <iostream>
#include <memory>
#include <vector>

using namespace nntile::autograd;

// Helper function to print tensor info
void print_tensor_info(const std::string& name, const AutogradTensor& tensor)
{
    std::cout << "Tensor: " << name << std::endl;
    std::cout << "  requires_grad: " << (tensor.requires_grad() ? "true" : "false") << std::endl;
    std::cout << "  has_grad_fn: " << (tensor.grad_fn() ? "true" : "false") << std::endl;
    if (tensor.grad_fn()) {
        std::cout << "  grad_fn: " << tensor.grad_fn()->name() << std::endl;
    }
    std::cout << std::endl;
}

int main()
{
    try {
        // Create input tensors
        AutogradTensor x(true);  // requires_grad = true
        AutogradTensor y(true);  // requires_grad = true
        AutogradTensor z(false); // requires_grad = false

        std::cout << "Initial tensors:" << std::endl;
        print_tensor_info("x", x);
        print_tensor_info("y", y);
        print_tensor_info("z", z);

        // Perform some operations
        auto a = add(2.0, x, 3.0, y);   // a = 2*x + 3*y
        auto b = add(1.0, a, 1.0, z);   // b = a + z
        auto c = add(1.0, b, 0.5, y);   // c = b + 0.5*y

        std::cout << "After operations:" << std::endl;
        print_tensor_info("a", a);
        print_tensor_info("b", b);
        print_tensor_info("c", c);

        // Visualize computation graph
        GraphVisualizer visualizer;
        std::cout << visualizer.visualize(c) << std::endl;

        // // Perform backward pass
        // std::cout << "Performing backward pass..." << std::endl;
        // c.backward();

        // std::cout << "After backward pass:" << std::endl;
        // print_tensor_info("x", x);
        // print_tensor_info("y", y);
        // print_tensor_info("z", z);
        // print_tensor_info("a", a);
        // print_tensor_info("b", b);
        // print_tensor_info("c", c);

        // // Test error handling
        // std::cout << "Testing error handling..." << std::endl;
        // try {
        //     // Try to add tensors with incompatible shapes (should fail in real implementation)
        //     AutogradTensor w(true);
        //     auto d = x.add(w);
        // } catch (const std::exception& e) {
        //     std::cout << "Caught expected exception: " << e.what() << std::endl;
        // }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
