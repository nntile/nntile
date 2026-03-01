/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/module/linear_manual.hh
 * Linear module with CUSTOM backward - demonstrates GradMode + wrap_with_module_op.
 *
 * PyTorch-like pattern: when a module has custom backward, forward runs in no_grad
 * so autograd ops don't build a graph; then output is wrapped in a single ModuleOp
 * whose backward is the module's build_backward.
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>
#include <vector>

#include <nntile/graph.hh>
#include <nntile/module/module.hh>

namespace nntile::module
{

//! Linear module with custom backward (overrides autograd of gemm/add_fiber).
//! Uses GradMode::Guard during forward so inner ops don't register producer;
//! then wrap_with_module_op attaches build_backward as the sole backward.
class LinearManual : public ModuleBase
{
private:
    graph::NNGraph::TensorNode* weight_tensor_ = nullptr;
    graph::NNGraph::TensorNode* bias_tensor_ = nullptr;
    graph::NNGraph::TensorNode* input_tensor_ = nullptr;
    graph::NNGraph::TensorNode* output_tensor_ = nullptr;
    Index input_dim_;
    Index output_dim_;
    graph::DataType dtype_;

public:
    LinearManual(
        graph::NNGraph& graph,
        const std::string& name,
        Index input_dim,
        Index output_dim,
        bool with_bias = false,
        graph::DataType dtype = graph::DataType::FP32);

    graph::NNGraph::TensorNode& build_forward(
        graph::NNGraph::TensorNode& input);

    std::vector<graph::NNGraph::TensorNode*> backward_inputs() const;

    void build_backward(const graph::NNGraph::OpNode* op);

    //! Forward: GradMode::Guard + build_forward + wrap_with_module_op
    graph::NNGraph::TensorNode& operator()(graph::NNGraph::TensorNode& input);

    std::string repr() const override;

    graph::NNGraph::TensorNode* weight_tensor() const { return weight_tensor_; }
    graph::NNGraph::TensorNode* bias_tensor() const { return bias_tensor_; }
    Index input_dim() const { return input_dim_; }
    Index output_dim() const { return output_dim_; }
};

} // namespace nntile::module
