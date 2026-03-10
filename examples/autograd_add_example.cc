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
 * After backward, converts TensorGraph -> TileGraph and executes on tiles.
 * Usage: autograd_add_example [x_requires_grad] [y_requires_grad]
 *   Each arg: 0 or 1 (default 0 if omitted).
 *
 * @version 1.1.0
 * */

#include <iostream>
#include <string>

#include <nntile/context.hh>
#include <nntile/graph.hh>

int main(int argc, char** argv)
{
    using namespace nntile::graph;
    namespace gt = nntile::graph::tensor;

    if(argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
    {
        std::cout << "Usage: autograd_add_example [x_requires_grad] [y_requires_grad]\n"
                     "  Diamond graph: w=x+y, v=w+y, z=v+w. Run z.backward().\n"
                     "  Converts TensorGraph to TileGraph and executes.\n"
                     "  Each arg: 0 or 1 (default 0 if omitted).\n";
        return 0;
    }

    bool x_requires_grad = (argc > 1 && std::string(argv[1]) != "0");
    bool y_requires_grad = (argc > 2 && std::string(argv[2]) != "0");

    nntile::Context context(1, 0, 0, "/tmp/nntile_ooc", 16777216, 0);

    NNGraph g("autograd_add_example");

    auto* x = g.tensor({2, 3}, "x", DataType::FP32, x_requires_grad);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32, y_requires_grad);

    auto* w = add(nntile::Scalar(1.0), x, nntile::Scalar(1.0), y, "w");
    auto* v = add(nntile::Scalar(1.0), w, nntile::Scalar(1.0), y, "v");
    auto* z = add(nntile::Scalar(1.0), v, nntile::Scalar(1.0), w, "z");

    std::cout << "=== Forward (diamond) ===" << std::endl;
    std::cout << "  w = x + y,  v = w + y,  z = v + w" << std::endl;
    std::cout << "  x_requires_grad=" << x_requires_grad
              << ", y_requires_grad=" << y_requires_grad << std::endl;

    auto [z_grad, _] = g.get_or_create_grad(z, "z_grad");
    gt::fill(nntile::Scalar(1.0), z_grad->data());
    z->backward();

    std::cout << "\n=== After z.backward() ===" << std::endl;
    std::cout << "  x.has_grad()=" << x->has_grad()
              << ", y.has_grad()=" << y->has_grad() << std::endl;

    size_t add_inplace_count = 0;
    for(const auto& op : g.tensor_graph().ops())
    {
        if(op->op_name() == "ADD_INPLACE")
        {
            ++add_inplace_count;
        }
    }
    std::cout << "  ADD_INPLACE count=" << add_inplace_count << std::endl;

    std::cout << "\n=== NNGraph structure ===" << std::endl;
    std::cout << g.to_string() << std::endl;

    // --- Mark inputs and outputs on TensorGraph for execution ---
    x->data()->mark_input(true);
    y->data()->mark_input(true);
    z->data()->mark_output(true);

    // --- Convert TensorGraph to TileGraph ---
    std::cout << "=== Converting TensorGraph -> TileGraph ===" << std::endl;
    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    std::cout << tile_graph.to_string() << std::endl;

    // --- Compile and execute TileGraph ---
    std::cout << "=== Compiling and executing TileGraph ===" << std::endl;
    TileGraph::Runtime tile_runtime(tile_graph);
    tile_runtime.compile();

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> y_data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    tile_runtime.bind_data("x", x_data);
    tile_runtime.bind_data("y", y_data);

    tile_runtime.execute();
    tile_runtime.wait();

    auto z_result = tile_runtime.get_output<float>("z");

    std::cout << "\n=== TileGraph execution results ===" << std::endl;
    std::cout << "  x = [";
    for(size_t i = 0; i < x_data.size(); ++i)
    {
        if(i > 0) std::cout << ", ";
        std::cout << x_data[i];
    }
    std::cout << "]" << std::endl;

    std::cout << "  y = [";
    for(size_t i = 0; i < y_data.size(); ++i)
    {
        if(i > 0) std::cout << ", ";
        std::cout << y_data[i];
    }
    std::cout << "]" << std::endl;

    std::cout << "  z = v + w = (w + y) + (x + y) = [";
    for(size_t i = 0; i < z_result.size(); ++i)
    {
        if(i > 0) std::cout << ", ";
        std::cout << z_result[i];
    }
    std::cout << "]" << std::endl;

    // Verify: w = x+y, v = w+y = x+2y, z = v+w = 2x+3y
    std::cout << "\n=== Verification (z = 2x + 3y) ===" << std::endl;
    bool correct = true;
    for(size_t i = 0; i < z_result.size(); ++i)
    {
        float expected = 2.0f * x_data[i] + 3.0f * y_data[i];
        if(std::abs(z_result[i] - expected) > 1e-5f)
        {
            std::cout << "  MISMATCH at " << i << ": got " << z_result[i]
                      << ", expected " << expected << std::endl;
            correct = false;
        }
    }
    if(correct)
    {
        std::cout << "  All values correct!" << std::endl;
    }

    std::cout << "\nAutograd add diamond example completed successfully."
              << std::endl;
    return 0;
}
