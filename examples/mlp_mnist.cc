/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/mlp_mnist.cc
 * Example of training an MLP-Mixer architecture on MNIST images
 *
 * @version 1.1.0
 * */

#include <nntile.hh>
#include <chrono>
#include <fstream>
#include <iostream>

using namespace nntile;
using T = fp32_t;

int main(int argc, char **argv)
{
    // Initialize StarPU and MPI
    //starpu_fxt_autostart_profiling(0);
    starpu::Config starpu(1, 0, 0);
    starpu_fxt_stop_profiling();
    starpu::init();
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    std::vector<int> mpi_grid = {2, 1};
    int mpi_root = 0;
    starpu_mpi_tag_t last_tag = 0;
    // Setup MNIST dataset for training
    Index n_pixels = 28 * 28; // MNIST contains 28x28 images
    Index n_images = 10000; // MNIST train contains 60k images
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
            ptr[i] = T{0};//static_cast<T>(mnist_single_buffer[i]);
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
    // Setup for the MLP encoder-decoder neural network
    Index n_mlp_encoder = 4; // Number of MLP layers in the encoder
    Index n_mlp_decoder = 4; // Number of MLP layers in the encoder
    std::vector<layer::MLP<T>> encoder, decoder;
    Index n_pixels_mlp = 4 * n_pixels;
    Index n_pixels_mlp_tile = 4 * n_pixels_tile;
    Index n_latent = 10; // Output size of encoder and inpout size of decoder
    tensor::TensorTraits mlp_linear1_traits({n_pixels_mlp, n_pixels},
            {n_pixels_mlp_tile, n_pixels_tile}),
        mlp_linear2_traits({n_pixels, n_pixels_mlp},
            {n_pixels_tile, n_pixels_mlp_tile}),
        mlp_gelu_traits({n_pixels_mlp, n_images},
            {n_pixels_mlp_tile, n_images_tile}),
        encoder_last_linear2_traits({n_latent, n_pixels_mlp},
            {n_latent, n_pixels_mlp_tile}),
        decoder_first_linear1_traits({n_pixels_mlp, n_latent},
            {n_pixels_mlp_tile, n_latent}),
        outputs_mid_traits({n_latent, n_images}, {n_latent, n_images_tile});
    std::vector<int> mlp_linear1_distr = tensor::distributions::block_cyclic(
            mlp_linear1_traits.grid.shape, mpi_grid, 0, mpi_size),
        mlp_linear2_distr = tensor::distributions::block_cyclic(
            mlp_linear2_traits.grid.shape, mpi_grid, 0, mpi_size),
        encoder_last_linear2_distr = tensor::distributions::block_cyclic(
            encoder_last_linear2_traits.grid.shape, mpi_grid, 0, mpi_size),
        decoder_first_linear1_distr = tensor::distributions::block_cyclic(
            decoder_first_linear1_traits.grid.shape, mpi_grid, 0, mpi_size),
        mlp_gelu_distr = tensor::distributions::block_cyclic(
            mlp_gelu_traits.grid.shape, mpi_grid, 0, mpi_size),
        outputs_mid_distr = tensor::distributions::block_cyclic(
            outputs_mid_traits.grid.shape, mpi_grid, 0, mpi_size);
    // Temporary outputs
    std::vector<tensor::Tensor<T>> outputs;
    outputs.reserve(n_mlp_encoder+n_mlp_decoder);
    // Set encoder
    encoder.reserve(n_mlp_encoder);
    unsigned long long seed = 0;
    T mean = 0.0, stddev = 1.0;
    std::vector<Index> zeros = {0, 0};
    for(Index i = 0; i < n_mlp_encoder-1; ++i)
    {
        encoder.emplace_back(mlp_linear1_traits, mlp_linear1_distr,
                mlp_linear2_traits, mlp_linear2_distr, mlp_gelu_traits,
                mlp_gelu_distr, last_tag);
        tensor::randn<T>(encoder[i].get_linear1().get_weight(), zeros,
                mlp_linear1_traits.shape, seed, mean, stddev);
        ++seed;
        tensor::randn<T>(encoder[i].get_linear2().get_weight(), zeros,
                mlp_linear2_traits.shape, seed, mean, stddev);
        ++seed;
        outputs.emplace_back(mnist_traits, mnist_distr, last_tag);
    }
    encoder.emplace_back(mlp_linear1_traits, mlp_linear1_distr,
            encoder_last_linear2_traits, encoder_last_linear2_distr,
            mlp_gelu_traits, mlp_gelu_distr, last_tag);
    tensor::randn<T>(encoder[n_mlp_encoder-1].get_linear1().get_weight(),
            zeros, mlp_linear1_traits.shape, seed, mean, stddev);
    ++seed;
    tensor::randn<T>(encoder[n_mlp_encoder-1].get_linear2().get_weight(),
            zeros, encoder_last_linear2_traits.shape, seed, mean, stddev);
    ++seed;
    outputs.emplace_back(outputs_mid_traits, outputs_mid_distr, last_tag);
    // Set decoder
    decoder.reserve(n_mlp_encoder);
    decoder.emplace_back(decoder_first_linear1_traits,
            decoder_first_linear1_distr,
            mlp_linear2_traits, mlp_linear2_distr, mlp_gelu_traits,
            mlp_gelu_distr, last_tag);
    tensor::randn<T>(decoder[0].get_linear1().get_weight(), zeros,
            decoder_first_linear1_traits.shape, seed, mean, stddev);
    ++seed;
    tensor::randn<T>(decoder[0].get_linear2().get_weight(), zeros,
            mlp_linear2_traits.shape, seed, mean, stddev);
    ++seed;
    outputs.emplace_back(mnist_traits, mnist_distr, last_tag);
    for(Index i = 1; i < n_mlp_decoder; ++i)
    {
        decoder.emplace_back(mlp_linear1_traits, mlp_linear1_distr,
                mlp_linear2_traits, mlp_linear2_distr, mlp_gelu_traits,
                mlp_gelu_distr, last_tag);
        tensor::randn<T>(decoder[i].get_linear1().get_weight(), zeros,
                mlp_linear1_traits.shape, seed, mean, stddev);
        ++seed;
        tensor::randn<T>(decoder[i].get_linear2().get_weight(), zeros,
                mlp_linear2_traits.shape, seed, mean, stddev);
        ++seed;
        outputs.emplace_back(mnist_traits, mnist_distr, last_tag);
    }
    // Do the forward-backward
    Index niter = 1;
    T learning_rate = 0.001;
    starpu_mpi_barrier(MPI_COMM_WORLD);
    starpu_fxt_start_profiling();
    for(Index i = 0; i < niter; ++i)
    {
        // Forward
        encoder[0].forward_async(mnist, outputs[0]);
        for(Index j = 1; j < n_mlp_encoder; ++j)
        {
            encoder[j].forward_async(outputs[j-1], outputs[j]);
        }
        for(Index j = 0; j < n_mlp_decoder-1; ++j)
        {
            Index k = j + n_mlp_encoder;
            decoder[j].forward_async(outputs[k-1], outputs[k]);
        }
        Index j = n_mlp_decoder - 1;
        Index k = j + n_mlp_encoder;
        tensor::copy_async<T>(mnist, outputs[k]);
        decoder[j].forward_async(outputs[k-1], T{-1}, outputs[k]);
        // Backward
        for(Index j = 0; j < n_mlp_decoder; ++j)
        {
            Index k = j + n_mlp_encoder;
            //decoder[j].backward_async(outputs[k-1], outputs[k]);
        }
    }
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
    starpu_mpi_barrier(MPI_COMM_WORLD);
    starpu_fxt_stop_profiling();
    return 0;
}
