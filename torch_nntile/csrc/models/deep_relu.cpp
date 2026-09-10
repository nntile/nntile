/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/deep_relu.cpp
 */

#include <torch_nntile/models/deep_relu.hh>

#include "nntile_gemm.h"
#include "nntile_nn_classic.h"

#include <stdexcept>

namespace torch_nntile::models
{

namespace
{

struct BiasFreeLinearImpl : torch::nn::Module
{
    torch::Tensor weight;

    BiasFreeLinearImpl(int64_t in_features, int64_t out_features)
    {
        weight = register_parameter(
            "weight",
            torch::empty({out_features, in_features}));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return torch_nntile::gemm(
            x,
            weight,
            1,
            0,
            false,
            true);
    }
};

TORCH_MODULE(BiasFreeLinear);

struct ReluImpl : torch::nn::Module
{
    torch::Tensor forward(torch::Tensor x)
    {
        return torch_nntile::nn_classic::relu(x);
    }
};

TORCH_MODULE(Relu);

} // namespace

DeepReLUImpl::DeepReLUImpl(
    int64_t input_dim_,
    int64_t hidden_dim_,
    int64_t output_dim_,
    int64_t depth_) :
    input_dim(input_dim_),
    hidden_dim(hidden_dim_),
    output_dim(output_dim_),
    depth(depth_)
{
    if (depth < 1)
    {
        throw std::invalid_argument("DeepReLU: depth must be >= 1");
    }
    torch::nn::Sequential seq;
    int64_t in_features = input_dim;
    int64_t out_features = (depth == 1) ? output_dim : hidden_dim;
    seq->push_back(BiasFreeLinear(in_features, out_features));
    for (int64_t i = 1; i < depth; ++i)
    {
        seq->push_back(Relu());
        in_features = hidden_dim;
        out_features =
            (i == depth - 1) ? output_dim : hidden_dim;
        seq->push_back(BiasFreeLinear(in_features, out_features));
    }
    net = register_module("net", seq);
}

torch::Tensor DeepReLUImpl::forward(torch::Tensor x)
{
    return net->forward(x);
}

torch::nn::ModuleHolder<DeepReLUImpl> DeepReLUImpl::tiny()
{
    return DeepReLU(128, 256, 10, 5);
}

torch::nn::ModuleHolder<DeepReLUImpl> DeepReLUImpl::mnist(
    int64_t hidden_dim,
    int64_t depth)
{
    return DeepReLU(28 * 28, hidden_dim, 10, depth);
}

} // namespace torch_nntile::models
