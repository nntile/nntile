/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/mlp_mixer.cpp
 * MLP-Mixer — port of main ``nntile.torch_models.mlp_mixer``.
 *
 * Side L: ``gemm(x, W, ndim=1, trans_b)``. Side R: ``gemm(W, x, ndim=1)``.
 * GAP: ``sum_slice`` (old ``nntile.layer.gap.GAP`` without side-R transpose).
 */

#include <torch_nntile/models/mlp_mixer.hh>

#include "nntile_gemm.h"
#include "nntile_sum_slice.h"

#include <cmath>
#include <stdexcept>

namespace torch_nntile::models
{

namespace
{

torch::Tensor linear_last_dim(
    torch::Tensor const &x,
    torch::Tensor const &weight)
{
    return gemm(
        x,
        weight,
        /*ndim=*/1,
        /*batch_ndim=*/0,
        /*trans_a=*/false,
        /*trans_b=*/true);
}

torch::Tensor linear_leading_dim(
    torch::Tensor const &weight,
    torch::Tensor const &x)
{
    // ``W [out, in] @ x [in, ...]`` → ``[out, ...]``.
    return gemm(
        weight,
        x,
        /*ndim=*/1,
        /*batch_ndim=*/0,
        /*trans_a=*/false,
        /*trans_b=*/false);
}

} // namespace

MixerMlpImpl::MixerMlpImpl(char side_, int64_t dim_) :
    side(side_),
    dim(dim_)
{
    if (side != 'L' && side != 'R')
    {
        throw std::invalid_argument("MixerMlp: side must be 'L' or 'R'");
    }
    if (dim <= 0)
    {
        throw std::invalid_argument("MixerMlp: dim must be positive");
    }
    fc1_weight = register_parameter(
        "fc1_weight",
        torch::empty({4 * dim, dim}));
    fc2_weight = register_parameter(
        "fc2_weight",
        torch::empty({dim, 4 * dim}));
    torch::nn::init::kaiming_uniform_(fc1_weight, std::sqrt(5.0));
    torch::nn::init::kaiming_uniform_(fc2_weight, std::sqrt(5.0));
}

torch::Tensor MixerMlpImpl::forward(torch::Tensor x)
{
    if (side == 'R')
    {
        auto h = linear_leading_dim(fc1_weight, x);
        h = torch::gelu(h);
        return linear_leading_dim(fc2_weight, h);
    }
    auto h = linear_last_dim(x, fc1_weight);
    h = torch::gelu(h);
    return linear_last_dim(h, fc2_weight);
}

MixerBlockImpl::MixerBlockImpl(
    int64_t channel_dim,
    int64_t patch_dim,
    double eps)
{
    norm_1 = register_module(
        "norm_1",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({patch_dim}).eps(eps)));
    mlp_1 = register_module("mlp_1", MixerMlp('R', channel_dim));
    norm_2 = register_module(
        "norm_2",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({patch_dim}).eps(eps)));
    mlp_2 = register_module("mlp_2", MixerMlp('L', patch_dim));
}

torch::Tensor MixerBlockImpl::forward(torch::Tensor x)
{
    auto y = mlp_1->forward(norm_1->forward(x)) + x;
    return mlp_2->forward(norm_2->forward(y)) + y;
}

MlpMixerImpl::MlpMixerImpl(MlpMixerConfig cfg) : config(std::move(cfg))
{
    if (config.num_mixer_layers < 1)
    {
        throw std::invalid_argument(
            "MlpMixer: num_mixer_layers must be >= 1");
    }
    stem_weight = register_parameter(
        "stem_weight",
        torch::empty(
            {config.projected_patch_dim, config.init_patch_dim}));
    torch::nn::ModuleList list;
    for (int64_t i = 0; i < config.num_mixer_layers; ++i)
    {
        list->push_back(
            MixerBlock(
                config.channel_dim,
                config.projected_patch_dim,
                config.layer_norm_epsilon));
    }
    blocks = register_module("blocks", list);
    classifier_weight = register_parameter(
        "classifier_weight",
        torch::empty({config.n_classes, config.projected_patch_dim}));
    torch::nn::init::kaiming_uniform_(stem_weight, std::sqrt(5.0));
    torch::nn::init::kaiming_uniform_(classifier_weight, std::sqrt(5.0));
}

torch::Tensor MlpMixerImpl::forward(torch::Tensor x)
{
    auto h = linear_last_dim(x, stem_weight);
    for (auto &module : *blocks)
    {
        h = module->as<MixerBlockImpl>()->forward(h);
    }
    // Old wrappers: sum_slice(1/P, h, axis=0) → [B, D]; then transpose for
    // side-R Linear. Torch layout keeps [B, D] + side-L classifier gemm.
    auto pooled = torch_nntile::gap(h);
    return linear_last_dim(pooled, classifier_weight);
}

} // namespace torch_nntile::models
