/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/mixer.cc
 * This is an example of a an MLP-Mixer architecture, only a forward pass
 *
 * @version 1.1.0
 * */

#include <nntile.hh>
#include <cmath>
#include <chrono>

using namespace nntile;
using T = fp32_t;

Starpu starpu;

int main(int argc, char **argv)
{
    // Inputs are 224x224 images, that are broken into PxP (P=14) patches.
    // Mixer parameters
    Index n_images = 1024; // single batch of images
    Index n_patch_pixels = 14 * 14;
    Index n_patches = 256; // Parameter S (number of patches, S=HW/P^2)
    Index n_patches_tile = 256;
    Index n_channels = 1280; // Parameter C (hidden dimension)
    Index n_channels_tile = 320;
    // pre-train 4096, fine-tune 512 (but we do not use this value)
    Index n_mlp_S = 640; // MLP size for patches
    Index n_mlp_S_tile = 640;//320;
    Index n_mlp_C = 5120; // MLP size for hidden dimension
    Index n_mlp_C_tile = 1024;//512;
    Index n_mixer_layers = 1; // 32; //
    T eps = std::sqrt(1e-5);
    // NNTile parameters
    Index n_images_tile = 512;
    // Tensor with input images
    std::vector<Index> input_images_shape = {n_patch_pixels, n_patches,
        n_images};
    std::vector<Index> input_images_basetile = {n_patch_pixels, n_patches_tile,
        n_images_tile};
    Tensor<T> input_images(input_images_shape, input_images_basetile);
    // Projector of PxP patches into hidden dimensions
    std::vector<Index> projector_shape = {n_patch_pixels, n_channels},
        projector_basetile = {n_patch_pixels, n_channels_tile};
    Tensor<T> projector(projector_shape, projector_basetile);
    // Tensor traits of input and output of each Mixer layer
    std::vector<Index> mixer_x_shape = {n_patches, n_images,
        n_channels};
    std::vector<Index> mixer_x_basetile = {n_patches_tile, n_images_tile,
        n_channels_tile};
    TensorTraits mixer_x_traits(mixer_x_shape, mixer_x_basetile);
    // Inputs and outputs of Mixer layers
    std::vector<Tensor<T>> mixer_x;
    mixer_x.reserve(n_mixer_layers+1);
    for(Index i = 0; i <= n_mixer_layers; ++i)
    {
        mixer_x.emplace_back(mixer_x_traits);
    }
    // Temporary for layer normalization
    Tensor<T> mixer_x_norm(mixer_x_traits);
    // Shape of linear layers of each MLP1 layer
    std::vector<Index> mixer_mlp1_fc1_shape = {n_mlp_S, n_patches},
        mixer_mlp1_fc1_basetile = {n_mlp_S_tile, n_patches_tile},
        mixer_mlp1_fc2_shape = {n_patches, n_mlp_S},
        mixer_mlp1_fc2_basetile = {n_patches_tile, n_mlp_S_tile};
    TensorTraits mixer_mlp1_fc1_traits(mixer_mlp1_fc1_shape,
            mixer_mlp1_fc1_basetile);
    TensorTraits mixer_mlp1_fc2_traits(mixer_mlp1_fc2_shape,
            mixer_mlp1_fc2_basetile);
    // MLP1 linear layers
    std::vector<Tensor<T>> mixer_mlp1_fc1, mixer_mlp1_fc2;
    mixer_mlp1_fc1.reserve(n_mixer_layers);
    mixer_mlp1_fc2.reserve(n_mixer_layers);
    for(Index i = 0; i < n_mixer_layers; ++i)
    {
        mixer_mlp1_fc1.emplace_back(mixer_mlp1_fc1_traits);
        mixer_mlp1_fc2.emplace_back(mixer_mlp1_fc2_traits);
    }
    // Shape of GeLU layer of each MLP1 layer
    std::vector<Index> mixer_x_mlp1_gelu_shape = {n_mlp_S, n_images,
        n_channels};
    std::vector<Index> mixer_x_mlp1_gelu_basetile = {n_mlp_S_tile,
        n_images_tile, n_channels_tile};
    TensorTraits mixer_x_mlp1_gelu_traits(mixer_x_mlp1_gelu_shape,
            mixer_x_mlp1_gelu_basetile);
    // Temporary for MLP1 GeLU layers
    Tensor<T> mixer_x_mlp1_gelu(mixer_x_mlp1_gelu_traits);
    // Shape of linear layers of each MLP2 layer
    std::vector<Index> mixer_mlp2_fc1_shape = {n_channels, n_mlp_C},
        mixer_mlp2_fc1_basetile = {n_channels_tile, n_mlp_C_tile},
        mixer_mlp2_fc2_shape = {n_mlp_C, n_channels},
        mixer_mlp2_fc2_basetile = {n_mlp_C_tile, n_channels_tile};
    TensorTraits mixer_mlp2_fc1_traits(mixer_mlp2_fc1_shape,
            mixer_mlp2_fc1_basetile);
    TensorTraits mixer_mlp2_fc2_traits(mixer_mlp2_fc2_shape,
            mixer_mlp2_fc2_basetile);
    // MLP2 linear layers
    std::vector<Tensor<T>> mixer_mlp2_fc1, mixer_mlp2_fc2;
    mixer_mlp2_fc1.reserve(n_mixer_layers);
    mixer_mlp2_fc2.reserve(n_mixer_layers);
    for(Index i = 0; i < n_mixer_layers; ++i)
    {
        mixer_mlp2_fc1.emplace_back(mixer_mlp2_fc1_traits);
        mixer_mlp2_fc2.emplace_back(mixer_mlp2_fc2_traits);
    }
    // Shape GeLU layer of each MLP2 layer
    std::vector<Index> mixer_x_mlp2_gelu_shape = {n_patches, n_images,
        n_mlp_C};
    std::vector<Index> mixer_x_mlp2_gelu_basetile = {n_patches_tile,
        n_images_tile, n_mlp_C_tile};
    TensorTraits mixer_x_mlp2_gelu_traits(mixer_x_mlp2_gelu_shape,
            mixer_x_mlp2_gelu_basetile);
    // Temporary for MLP2 GeLU layers
    Tensor<T> mixer_x_mlp2_gelu(mixer_x_mlp2_gelu_traits);
    // Temporary arrays for first layer normalization
    Tensor<T> sum_ssq1({3, n_patches, n_images},
            {3, n_patches_tile, n_images_tile}),
        avg_dev1({2, n_patches, n_images},
                {2, n_patches_tile, n_images_tile});
    // Temporary arrays for second layer normalization
    Tensor<T> sum_ssq2({3, n_images, n_channels},
            {3, n_images_tile, n_channels_tile}),
        avg_dev2({2, n_images, n_channels},
                {2, n_images_tile, n_channels_tile});
    // Randomly init inputs
    unsigned long long seed_input = 100ULL, seed_project = 101ULL;
    std::vector<unsigned long long> seed_mlp1_fc1(n_mixer_layers),
        seed_mlp1_fc2(n_mixer_layers), seed_mlp2_fc1(n_mixer_layers),
        seed_mlp2_fc2(n_mixer_layers);
    seed_mlp1_fc1[0] = 102ULL;
    seed_mlp1_fc2[0] = seed_mlp1_fc1[0] + n_mixer_layers;
    seed_mlp2_fc1[0] = seed_mlp1_fc2[0] + n_mixer_layers;
    seed_mlp2_fc2[0] = seed_mlp2_fc1[0] + n_mixer_layers;
    for(Index i = 1; i < n_mixer_layers; ++i)
    {
        seed_mlp1_fc1[i] = seed_mlp1_fc1[0] + i;
        seed_mlp1_fc2[i] = seed_mlp1_fc2[0] + i;
        seed_mlp2_fc1[i] = seed_mlp2_fc1[0] + i;
        seed_mlp2_fc2[i] = seed_mlp2_fc2[0] + i;
    }
    // Asynchronous generation
    randn_async(input_images, seed_input);
    randn_async(projector, seed_project);
    for(Index i = 0; i < n_mixer_layers; ++i)
    {
        randn_async(mixer_mlp1_fc1[i], seed_mlp1_fc1[i]);
        randn_async(mixer_mlp1_fc2[i], seed_mlp1_fc2[i]);
        randn_async(mixer_mlp2_fc1[i], seed_mlp2_fc1[i]);
        randn_async(mixer_mlp2_fc2[i], seed_mlp2_fc2[i]);
    }
    starpu_task_wait_for_all();
    // Project patches into hidden dimensions
    gemm(T{1}, TransOp::Trans, input_images, TransOp::NoTrans, projector,
            T{0}, mixer_x[0], 1);
    // Flush profiling data
    std::chrono::steady_clock clock;
    starpu_profiling_init();
    starpu_profiling_worker_helper_display_summary();
    starpu_fxt_start_profiling();
    auto start = clock.now();
    // Apply all mixer layers
    for(Index i = 0; i < n_mixer_layers; ++i)
    {
        // Skip connection
        copy_async(mixer_x[i], mixer_x[i+1]);
        // Layer normalization
        copy_async(mixer_x[i], mixer_x_norm);
        mixer_x[i].wont_use();
        norm_sum_ssq_async(mixer_x_norm, sum_ssq1, 2);
        norm_avg_dev_async(sum_ssq1, avg_dev1, n_channels, eps);
        sum_ssq1.invalidate_submit();
        bias2_async(avg_dev1, mixer_x_norm, 2);
        avg_dev1.invalidate_submit();
        // Apply MLP1
        gemm_async(T{1}, TransOp::NoTrans, mixer_mlp1_fc1[i], TransOp::NoTrans,
                mixer_x_norm, T{0}, mixer_x_mlp1_gelu);
        mixer_x_norm.invalidate_submit();
        // GeLU is too slow, use ReLU instead
        relu_async(mixer_x_mlp1_gelu);
        gemm_async(T{1}, TransOp::NoTrans, mixer_mlp1_fc2[i], TransOp::NoTrans,
                mixer_x_mlp1_gelu, T{1}, mixer_x[i+1]);
        mixer_x_mlp1_gelu.invalidate_submit();
        // Skip connection is already inplace
        // Layer normalization
        copy_async(mixer_x[i+1], mixer_x_norm);
        norm_sum_ssq_async(mixer_x_norm, sum_ssq2, 0);
        norm_avg_dev_async(sum_ssq2, avg_dev2, n_patches, eps);
        sum_ssq2.invalidate_submit();
        bias2_async(avg_dev2, mixer_x_norm, 0);
        avg_dev2.invalidate_submit();
        // Apply MLP2
        gemm_async(T{1}, TransOp::NoTrans, mixer_x_norm, TransOp::NoTrans,
                mixer_mlp2_fc1[i], T{0}, mixer_x_mlp2_gelu);
        mixer_x_norm.invalidate_submit();
        // GeLU is too slow, use ReLU instead
        relu_async(mixer_x_mlp2_gelu);
        gemm_async(T{1}, TransOp::NoTrans, mixer_x_mlp2_gelu,
                TransOp::NoTrans, mixer_mlp2_fc2[i], T{1}, mixer_x[i+1]);
        mixer_x_mlp2_gelu.invalidate_submit();
    }
    // The last layer is yet missing
    starpu_task_wait_for_all();
    auto end = clock.now();
    starpu_fxt_stop_profiling();
    starpu_profiling_worker_helper_display_summary();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Done in " << diff.count() << " seconds\n";
    return 0;
}
