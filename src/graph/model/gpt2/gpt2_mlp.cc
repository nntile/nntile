/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/gpt2/gpt2_mlp.cc
 * GPT2MLP implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/gpt2/gpt2_mlp.hh"
#include "nntile/graph/nn/ops/add_fiber.hh"
#include "nntile/graph/nn/ops/gemm.hh"

namespace nntile::model::gpt2
{

Gpt2MLP::Gpt2MLP(graph::NNGraph* graph,
                 const std::string& name,
                 const Gpt2Config& config,
                 graph::DataType dtype)
    : graph::module::Mlp(graph, name,
                         config.hidden_size,
                         config.intermediate_size,
                         config.hidden_size,
                         graph::module::ActivationType::GELUTANH,
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

graph::NNGraph::TensorNode* Gpt2MLP::forward(
    graph::NNGraph::TensorNode* input)
{
    graph::NNGraph::TensorNode* w1 = fc1().weight_tensor();
    graph::NNGraph::TensorNode* hidden =
        graph::gemm(w1, input, 1.0, true, false, 1, 0);
    hidden->set_name(tensor_name("fc1_out"));
    hidden = graph::add_fiber(1.0, fc1_bias_, 1.0, hidden, 0, 0);
    hidden = activation().forward(hidden);
    graph::NNGraph::TensorNode* w2 = fc2().weight_tensor();
    graph::NNGraph::TensorNode* out =
        graph::gemm(w2, hidden, 1.0, true, false, 1, 0);
    out = graph::add_fiber(1.0, fc2_bias_, 1.0, out, 0, 0);
    out->set_name(tensor_name("output"));
    return out;
}

std::string Gpt2MLP::repr() const
{
    return "Gpt2MLP(" + graph::module::Mlp::repr() + ")";
}

} // namespace nntile::model::gpt2
