/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/model/gpt2/gpt2_mlp.cc
 * GPT2MLP implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/model/gpt2/gpt2_mlp.hh"
#include "nntile/nn/ops/add_fiber.hh"
#include "nntile/nn/ops/gemm.hh"

namespace nntile::model::gpt2
{

Gpt2MLP::Gpt2MLP(NNGraph* graph,
                 const std::string& name,
                 const Gpt2Config& config,
                 DataType dtype)
    : module::Mlp(graph, name,
                         config.hidden_size,
                         config.intermediate_size,
                         config.hidden_size,
                         module::ActivationType::GELUTANH,
                         dtype)
{
    config.validate();

    fc1_bias_ = graph_->tensor({config.intermediate_size}, dtype, true);
    fc1_bias_->set_name(tensor_name("fc1.bias"));
    register_parameter("fc1.bias", fc1_bias_);

    fc2_bias_ = graph_->tensor({config.hidden_size}, dtype, true);
    fc2_bias_->set_name(tensor_name("fc2.bias"));
    register_parameter("fc2.bias", fc2_bias_);
}

NNGraph::TensorNode* Gpt2MLP::forward(
    NNGraph::TensorNode* input)
{
    NNGraph::TensorNode* w1 = fc1().weight_tensor();
    NNGraph::TensorNode* hidden =
        gemm(w1, input, 1.0, true, false, 1, 0);
    hidden->set_name(tensor_name("fc1_out"));
    hidden = add_fiber(1.0, fc1_bias_, 1.0, hidden, 0, 0);
    hidden = activation().forward(hidden);
    NNGraph::TensorNode* w2 = fc2().weight_tensor();
    NNGraph::TensorNode* out =
        gemm(w2, hidden, 1.0, true, false, 1, 0);
    out = add_fiber(1.0, fc2_bias_, 1.0, out, 0, 0);
    out->set_name(tensor_name("output"));
    return out;
}

std::string Gpt2MLP::repr() const
{
    return "Gpt2MLP(" + module::Mlp::repr() + ")";
}

} // namespace nntile::model::gpt2
