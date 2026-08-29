/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/mlp_mixer.hh
 * MLP-Mixer matching main ``nntile.torch_models.mlp_mixer``.
 */

#pragma once

#include <torch/torch.h>
#include <torch_nntile/classic_nn.hh>

#include <cstdint>

namespace torch_nntile::models
{

struct MlpMixerConfig
{
    int64_t channel_dim = 8;
    int64_t init_patch_dim = 4;
    int64_t projected_patch_dim = 4;
    int64_t num_mixer_layers = 2;
    int64_t n_classes = 3;
    double layer_norm_epsilon = 1e-5;
};

//! Bias-free expand-4 MLP; side ``L`` last-dim, ``R`` mix axis 0.
struct MixerMlpImpl : torch::nn::Module
{
    char side = 'L';
    int64_t dim = 0;
    torch::Tensor fc1_weight;
    torch::Tensor fc2_weight;

    MixerMlpImpl(char side_, int64_t dim_);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(MixerMlp);

struct MixerBlockImpl : torch::nn::Module
{
    nn_classic::LayerNorm norm_1{nullptr};
    MixerMlp mlp_1{nullptr};
    nn_classic::LayerNorm norm_2{nullptr};
    MixerMlp mlp_2{nullptr};

    MixerBlockImpl(
        int64_t channel_dim,
        int64_t patch_dim,
        double eps);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(MixerBlock);

struct MlpMixerImpl : torch::nn::Module
{
    MlpMixerConfig config;
    torch::Tensor stem_weight;
    torch::nn::ModuleList blocks{nullptr};
    torch::Tensor classifier_weight;

    explicit MlpMixerImpl(MlpMixerConfig cfg);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(MlpMixer);

} // namespace torch_nntile::models
