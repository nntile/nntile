/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/deep_relu.hh
 * Bias-free DeepReLU MLP for device=nntile.
 */

#pragma once

#include <torch/torch.h>

namespace torch_nntile::models
{

//! Bias-free Linear -> ReLU chain (matches Python DeepReLU).
struct DeepReLUImpl : torch::nn::Module
{
    int64_t input_dim = 0;
    int64_t hidden_dim = 0;
    int64_t output_dim = 0;
    int64_t depth = 0;
    torch::nn::Sequential net{nullptr};

    DeepReLUImpl(
        int64_t input_dim_,
        int64_t hidden_dim_,
        int64_t output_dim_,
        int64_t depth_);

    torch::Tensor forward(torch::Tensor x);

    static torch::nn::ModuleHolder<DeepReLUImpl> tiny();
    static torch::nn::ModuleHolder<DeepReLUImpl> mnist(
        int64_t hidden_dim = 256,
        int64_t depth = 5);
};

TORCH_MODULE(DeepReLU);

} // namespace torch_nntile::models
