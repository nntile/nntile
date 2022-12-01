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
    starpu::Config starpu(4, 0, 0);
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
    // Set the deep linear network
    Index n_linear = 4;
    std::vector<layer::Linear<T>> linear;
    std::vector<tensor::Tensor<T>> params, grads, tmps;
    tensor::TensorTraits
        tmp_traits({4*n_pixels, n_images}, {4*n_pixels_tile, n_images_tile}),
        w1_traits({4*n_pixels, n_pixels}, {4*n_pixels_tile, n_pixels_tile}),
        w2_traits({4*n_pixels, 4*n_pixels}, {4*n_pixels_tile, 4*n_pixels_tile}),
        w3_traits({n_pixels, 4*n_pixels}, {n_pixels_tile, 4*n_pixels_tile});
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
    T mean = 0.0, stddev = 1.0;
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
    Index n_iter = 10;
    for(Index iter = 0; iter < n_iter; ++iter)
    {
        tensor::copy_async<T>(mnist, tmps[0]);
        // Forward loop through layers
        for(Index i = 0; i < n_linear; ++i)
        {
            linear[i].forward_async(tmps[i], tmps[i+1]);
        }
        // Subtract input out of output for gradient
        tensor::axpy2_async<T>(-1.0, mnist, tmps[n_linear]);
        // Backward loop through layers
        for(Index i = n_linear-1; i >= 0; --i)
        {
            linear[i].backward_async(tmps[i], tmps[i+1], tmps[i]);
            // Update parameters
            tensor::axpy2_async<T>(-1e-5, grads[i], params[i]);
        }
    }
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
    starpu_mpi_barrier(MPI_COMM_WORLD);
    starpu_fxt_stop_profiling();
    return 0;
}

