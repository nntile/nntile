/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/classic_nn.hh
 * Classic nntile::kernel LayerNorm / Embedding modules.
 */

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace torch_nntile::nn_classic
{

struct LayerNormImpl : torch::nn::Module
{
    torch::Tensor weight;
    torch::Tensor bias;
    std::vector<int64_t> normalized_shape;
    double eps = 1e-5;

    LayerNormImpl(int64_t normalized_size, double eps_);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(LayerNorm);

struct EmbeddingImpl : torch::nn::Module
{
    torch::Tensor weight;

    EmbeddingImpl(int64_t num_embeddings, int64_t embedding_dim);
    torch::Tensor forward(torch::Tensor indices);
};

TORCH_MODULE(Embedding);

} // namespace torch_nntile::nn_classic
