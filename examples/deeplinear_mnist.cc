/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/deeplinear_mnist.cc
 * Example of training a DeepLinear network without activation function on
 * MNIST images
 *
 * @version 1.1.0
 * */

#include <nntile.hh>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cmath>

using namespace nntile;
using T = fp32_t;

int main(int argc, char **argv)
{
    // Initialize StarPU and MPI
    //starpu_fxt_autostart_profiling(0);
    starpu::Config starpu(4, 1, 1);
    starpu_fxt_stop_profiling();
    starpu::init();
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    // Parameters to be read with default values
    Index n_images = 10000; // MNIST train contains 60k images
    Index n_linear = 3; // Number of linear layers
    bool print_loss = true; // Print loss or not
    // Read MNIST train data filename
    if(argc == 1)
    {
        if(mpi_rank == 0)
        {
            std::cerr << "Please provide path to MNIST file "
                "train-images-idx3-ubyte\n";
        }
        exit(0);
    }
    // Read number of images to be used
    if(argc >= 3)
    {
        n_images = std::atoi(argv[2]);
        if(n_images <= 0 or n_images > 60000)
        {
            n_images = 10000;
        }
    }
    // Read number of linear layers
    if(argc >= 4)
    {
        n_linear = std::atoi(argv[3]);
        if(n_linear <= 0 or n_linear > 10)
        {
            n_linear = 3;
        }
    }
    // Read flag if we need to output loss
    if(argc >= 5)
    {
        print_loss = std::atoi(argv[4]);
    }
    // Proceed with other things
    std::vector<int> mpi_grid = {2, 1};
    int mpi_root = 0;
    starpu_mpi_tag_t last_tag = 0;
    // Setup MNIST dataset for training
    Index n_pixels = 28 * 28; // MNIST contains 28x28 images
    tensor::TensorTraits mnist_single_traits({n_pixels, n_images},
            {n_pixels, n_images});
    std::vector<int> distr_root = {mpi_root};
    tensor::Tensor<T> mnist_single(mnist_single_traits, distr_root, last_tag);
    if(mpi_rank == mpi_root)
    {
        // Read MNIST as byte-sized unsigned integers [0..255]
        std::ifstream f;
        f.open(argv[1]);
        std::size_t mnist_pixels = n_pixels * n_images;
        auto mnist_single_buffer = new unsigned char[mnist_pixels];
        std::cout << "Start reading " << mnist_pixels << " bytes\n";
        // Skip first 16 bytes
        f.seekg(16, std::ios_base::beg);
        f.read(reinterpret_cast<char *>(mnist_single_buffer), mnist_pixels);
        std::cout << "Read " << f.gcount() << " bytes from " << argv[1]
            << "\n";
        f.close();
        // Convert integers into floating point numbers
        auto tile = mnist_single.get_tile_handle(0);
        auto tile_local = tile.acquire(STARPU_W);
        T *ptr = reinterpret_cast<T *>(tile_local.get_ptr());
        std::cout << "MNIST SINGLE: " << mnist_single.get_tile_traits(0).nelems
            << " " << mnist_pixels << "\n";
        for(Index i = 0; i < mnist_pixels; ++i)
        {
            ptr[i] = static_cast<T>(mnist_single_buffer[i]);
        }
        tile_local.release();
        // Clear MNIST byte buffer
        delete[] mnist_single_buffer;
    }
    // Distribute MNIST input
    Index n_pixels_tile = 28 * 28;
    Index n_images_tile = 1024;
    tensor::TensorTraits mnist_traits({n_pixels, n_images},
            {n_pixels_tile, n_images_tile});
    std::vector<int> mnist_distr = tensor::distributions::block_cyclic(
            mnist_traits.grid.shape, mpi_grid, 0, mpi_size);
    tensor::Tensor<T> mnist(mnist_traits, mnist_distr, last_tag);
    tensor::scatter<T>(mnist_single, mnist);
    // Set the deep linear network
    std::vector<layer::Linear<T>> linear;
    std::vector<tensor::Tensor<T>> params, grads, tmps;
//    tensor::TensorTraits
//        tmp_traits({4*n_pixels, n_images}, {4*n_pixels_tile, n_images_tile}),
//        w1_traits({4*n_pixels, n_pixels}, {4*n_pixels_tile, n_pixels_tile}),
//        w2_traits({4*n_pixels, 4*n_pixels}, {4*n_pixels_tile, 4*n_pixels_tile}),
//        w3_traits({n_pixels, 4*n_pixels}, {n_pixels_tile, 4*n_pixels_tile});
    tensor::TensorTraits
        tmp_traits({n_pixels, n_images}, {n_pixels_tile, n_images_tile}),
        w1_traits({n_pixels, n_pixels}, {n_pixels_tile, n_pixels_tile}),
        w2_traits({n_pixels, n_pixels}, {n_pixels_tile, n_pixels_tile}),
        w3_traits({n_pixels, n_pixels}, {n_pixels_tile, n_pixels_tile});
    std::vector<int> tmp_distr = tensor::distributions::block_cyclic(
            tmp_traits.grid.shape, mpi_grid, 0, mpi_size),
        w1_distr = tensor::distributions::block_cyclic(
            w1_traits.grid.shape, mpi_grid, 0, mpi_size),
        w2_distr = tensor::distributions::block_cyclic(
            w2_traits.grid.shape, mpi_grid, 0, mpi_size),
        w3_distr = tensor::distributions::block_cyclic(
            w3_traits.grid.shape, mpi_grid, 0, mpi_size);
    params.reserve(n_linear);
    grads.reserve(n_linear);
    tmps.reserve(n_linear+1);
    linear.reserve(n_linear);
    params.emplace_back(w1_traits, w1_distr, last_tag);
    grads.emplace_back(w1_traits, w1_distr, last_tag);
    tmps.emplace_back(mnist_traits, tmp_distr, last_tag);
    tmps.emplace_back(tmp_traits, tmp_distr, last_tag);
    linear.emplace_back(mnist_traits, tmp_traits, params[0], grads[0]);
    for(Index i = 1; i < n_linear-1; ++i)
    {
        params.emplace_back(w2_traits, w2_distr, last_tag);
        grads.emplace_back(w2_traits, w2_distr, last_tag);
        tmps.emplace_back(tmp_traits, tmp_distr, last_tag);
        linear.emplace_back(tmp_traits, tmp_traits, params[i], grads[i]);
    }
    params.emplace_back(w3_traits, w3_distr, last_tag);
    grads.emplace_back(w3_traits, w3_distr, last_tag);
    tmps.emplace_back(mnist_traits, mnist_distr, last_tag);
    linear.emplace_back(tmp_traits, mnist_traits, params[n_linear-1],
            grads[n_linear-1]);
    // Init linear layers
    T mean = 0.0, stddev = 1.0 / std::sqrt(T(n_pixels));
    unsigned long long seed = -1;
    for(Index i = 0; i < n_linear; ++i)
    {
        tensor::randn_async<T>(params[i], {0, 0}, params[i].shape, seed, mean,
                stddev);
        --seed;
    }
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
    // Iterate
    Index n_iter = 100;
    tensor::TensorTraits loss_traits({}, {}),
                 tmp_loss_traits(mnist_traits.grid.shape, {1, 1});
    tensor::Tensor<T> loss(loss_traits, distr_root, last_tag);
    tensor::Tensor<T> tmp_loss(tmp_loss_traits, mnist_distr, last_tag);
    // Start timer
    starpu_mpi_barrier(MPI_COMM_WORLD);
    double time_elapsed = -MPI_Wtime();
    for(Index iter = 0; iter < n_iter; ++iter)
    {
        tensor::copy_async<T>(mnist, tmps[0]);
        // Forward loop through layers
        for(Index i = 0; i < n_linear; ++i)
        {
            linear[i].forward_async(tmps[i], tmps[i+1]);
        }
        // Subtract input out of output for gradient
        tensor::axpy_async<T>(-1.0, mnist, tmps[n_linear]);
        // Get loss
        tensor::nrm2_async<T>(tmps[n_linear], loss, tmp_loss);
        // Backward loop through layers
        for(Index i = n_linear-1; i >= 0; --i)
        {
            linear[i].backward_async(tmps[i], tmps[i+1], tmps[i]);
            // Update parameters
            tensor::axpy_async<T>(-1e-12, grads[i], params[i]);
        }
        if(print_loss and mpi_rank == mpi_root)
        {
            auto loss_local = loss.get_tile(0).acquire(STARPU_R);
            std::cout << "iter=" << iter << " loss=" << loss_local[0] << "\n";
            loss_local.release();
        }
    }
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
    starpu_mpi_barrier(MPI_COMM_WORLD);
    time_elapsed += MPI_Wtime();
    starpu_fxt_stop_profiling();
    std::cout << "Training time: " << time_elapsed << " seconds\n";
//    if(mpi_rank == mpi_root)
//    {
//        auto loss_local = loss.get_tile(0).acquire(STARPU_R);
//        std::cout << "Final loss=" << loss_local[0] << "\n";
//        loss_local.release();
//    }
    return 0;
}
