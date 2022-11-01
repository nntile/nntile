#include <nntile.hh>
#include <chrono>
#include <fstream>
#include <iostream>

using namespace nntile;
using T = fp32_t;

int main(int argc, char **argv)
{
    // Initialize StarPU and MPI
    starpu::Config starpu(-1, -1, -1);
    starpu::init();
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    std::vector<int> mpi_grid = {2, 2};
    int mpi_root = 0;
    starpu_mpi_tag_t last_tag = 0;
    // Setup MNIST dataset for training
    Index n_pixels = 28 * 28; // MNIST contains 28x28 images
    Index n_images = 1000; // MNIST train contains 60k images
    std::ifstream f;
    f.open(argv[1]);
    std::size_t mnist_pixels = n_pixels * n_images;
    auto mnist_single_buffer = new unsigned char[mnist_pixels];
    std::cout << "Start reading " << mnist_pixels << " bytes\n";
    // Skip first 16 bytes
    f.seekg(16, std::ios_base::beg);
    f.read(reinterpret_cast<char *>(mnist_single_buffer), mnist_pixels);
    std::cout << "Read " << f.gcount() << " bytes from " << argv[1] << "\n";
    f.close();
    // Convert MNIST byte-size unsigned integers into floating point numbers
    tensor::TensorTraits mnist_single_traits({n_pixels, n_images},
            {n_pixels, n_images});
    std::vector<int> distr_root = {mpi_root};
    tensor::Tensor<T> mnist_single(mnist_single_traits, distr_root, last_tag);
    auto tile = mnist_single.get_tile(0);
    auto tile_local = tile.acquire(STARPU_W);
    for(Index i = 0; i < mnist_pixels; ++i)
    {
        tile_local[i] = static_cast<T>(mnist_single_buffer[i]);
    }
    tile_local.release();
    // Clear MNIST byte buffer
    delete[] mnist_single_buffer;
    // Distribute MNIST input
    Index n_images_tile = 500;
    Index n_pixels_tile = 14 * 28;
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
    for(Index i = 0; i < niter; ++i)
    {
        // Forward
        encoder[0].forward_async(mnist, outputs[0]);
        for(Index j = 1; j < n_mlp_encoder; ++j)
        {
            encoder[j].forward_async(outputs[j-1], outputs[j]);
        }
        for(Index j = 0; j < n_mlp_decoder; ++j)
        {
            Index k = j + n_mlp_encoder;
            decoder[j].forward_async(outputs[k-1], outputs[k]);
        }
        // Overwrite one last gemm to get antigradient
        tensor::copy<T>(mnist, outputs[n_mlp_encoder+n_mlp_decoder-1]);
        constexpr T one = 1, mone = -1;
        constexpr TransOp opN(TransOp::NoTrans);
        tensor::gemm<T>(mone, opN,
                decoder[n_mlp_decoder-1].get_linear2().get_weight(),
                opN, decoder[n_mlp_decoder-1].get_gelu(), one,
                outputs[n_mlp_encoder+n_mlp_decoder-1], 1);
        // Backward

    }
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
    return 0;
}

