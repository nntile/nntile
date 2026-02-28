/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/module/linear_manual.cc
 * LinearManual - module with custom backward (GradMode + wrap_with_module_op).
 *
 * @version 1.1.0
 * */

#include "nntile/module/linear_manual.hh"
#include "nntile/graph/logical/gemm.hh"
#include "nntile/graph/logical/sum_fiber.hh"

#include <stdexcept>

namespace nntile::module
{

namespace
{
constexpr Scalar GEMM_ALPHA = 1.0;
constexpr Scalar GEMM_BETA_OVERWRITE = 0.0;
constexpr Scalar GEMM_BETA_ACCUMULATE = 1.0;
constexpr Index GEMM_NDIM_MATRIX = 1;
constexpr Index NO_BATCH_DIM = 0;
constexpr bool NO_TRANSPOSE = false;
constexpr bool TRANSPOSE = true;
constexpr Scalar ADD_FIBER_ALPHA = 1.0;
constexpr Scalar ADD_FIBER_BETA = 1.0;
constexpr Index SUM_FIBER_BATCH_NDIM = 0;
constexpr int SUM_FIBER_REDUX_NONE = 0;
constexpr Scalar SUM_FIBER_ALPHA = 1.0;
constexpr Scalar SUM_FIBER_BETA_OVERWRITE = 0.0;
constexpr Scalar SUM_FIBER_BETA_ACCUMULATE = 1.0;
} // namespace

LinearManual::LinearManual(
    graph::NNGraph& graph,
    const std::string& name,
    Index input_dim,
    Index output_dim,
    bool with_bias,
    graph::DataType dtype)
    : Module(graph, name)
    , input_dim_(input_dim)
    , output_dim_(output_dim)
    , dtype_(dtype)
{
    weight_tensor_ = graph_.tensor(
        {input_dim_, output_dim_},
        tensor_name("weight"),
        dtype_,
        true);
    register_parameter("weight", weight_tensor_);
    if(with_bias)
    {
        bias_tensor_ = graph_.tensor(
            {output_dim_},
            tensor_name("bias"),
            dtype_,
            true);
        register_parameter("bias", bias_tensor_);
    }
}

graph::NNGraph::TensorNode& LinearManual::build_forward(
    graph::NNGraph::TensorNode& input)
{
    return forward(input);
}

graph::NNGraph::TensorNode& LinearManual::forward_impl(
    graph::NNGraph::TensorNode& input)
{
    if(input.ndim() < 1)
    {
        throw std::invalid_argument(
            "LinearManual::forward_impl: input must have at least one dimension");
    }
    if(input.shape().back() != input_dim_)
    {
        throw std::invalid_argument(
            "LinearManual::forward_impl: input feature dim mismatch");
    }

    input_tensor_ = &input;

    const std::string gemm_name = bias_tensor_ != nullptr
        ? tensor_name("gemm_output")
        : tensor_name("output");
    graph::NNGraph::TensorNode* gemm_out = graph::gemm(
        &input,
        weight_tensor_,
        gemm_name,
        GEMM_ALPHA,
        NO_TRANSPOSE,
        NO_TRANSPOSE,
        GEMM_NDIM_MATRIX,
        NO_BATCH_DIM);

    if(bias_tensor_ != nullptr)
    {
        const Index feature_axis = gemm_out->ndim() - 1;
        output_tensor_ = graph::add_fiber(
            ADD_FIBER_ALPHA,
            bias_tensor_,
            ADD_FIBER_BETA,
            gemm_out,
            tensor_name("output"),
            feature_axis,
            NO_BATCH_DIM);
    }
    else
    {
        output_tensor_ = gemm_out;
    }

    return *output_tensor_;
}

std::vector<graph::NNGraph::TensorNode*> LinearManual::backward_inputs() const
{
    std::vector<graph::NNGraph::TensorNode*> inputs = {input_tensor_,
                                                       weight_tensor_};
    if(bias_tensor_ != nullptr)
    {
        inputs.push_back(bias_tensor_);
    }
    return inputs;
}

void LinearManual::build_backward(const graph::NNGraph::OpNode* op)
{
    graph::NNGraph::TensorNode* grad_output = op->output()->grad();
    if(grad_output == nullptr)
    {
        return;
    }

    const auto& inputs = op->inputs();
    if(inputs.size() < 2)
    {
        return;
    }
    graph::NNGraph::TensorNode* input_nn = inputs[0];
    graph::NNGraph::TensorNode* weight_nn = inputs[1];
    graph::NNGraph::TensorNode* bias_nn =
        inputs.size() >= 3 ? inputs[2] : nullptr;

    // grad_weight += input^T @ grad_output
    if(weight_nn != nullptr && graph_.requires_grad(weight_nn))
    {
        bool first = graph_.is_first_grad(weight_nn);
        graph::NNGraph::TensorNode* grad_weight =
            graph_.get_or_create_grad(weight_nn, grad_name("weight"));
        Scalar beta = first ? GEMM_BETA_OVERWRITE : GEMM_BETA_ACCUMULATE;
        graph::gemm(
            input_nn->data(),
            grad_output->data(),
            grad_weight->data(),
            GEMM_ALPHA,
            beta,
            TRANSPOSE,
            NO_TRANSPOSE,
            GEMM_NDIM_MATRIX,
            NO_BATCH_DIM);
    }

    // grad_bias += sum_fiber(grad_output)
    if(bias_nn != nullptr && graph_.requires_grad(bias_nn))
    {
        bool first = graph_.is_first_grad(bias_nn);
        graph::NNGraph::TensorNode* grad_bias =
            graph_.get_or_create_grad(bias_nn, grad_name("bias"));
        Scalar beta = first ? SUM_FIBER_BETA_OVERWRITE : SUM_FIBER_BETA_ACCUMULATE;
        const Index feature_axis = grad_output->ndim() - 1;
        graph::sum_fiber(
            grad_output->data(),
            grad_bias->data(),
            feature_axis,
            SUM_FIBER_BATCH_NDIM,
            SUM_FIBER_REDUX_NONE,
            SUM_FIBER_ALPHA,
            beta);
    }

    // grad_input += grad_output @ weight^T
    if(input_nn != nullptr && graph_.requires_grad(input_nn))
    {
        bool first = graph_.is_first_grad(input_nn);
        graph::NNGraph::TensorNode* grad_input =
            graph_.get_or_create_grad(input_nn, input_nn->name() + "_grad");
        Scalar beta = first ? GEMM_BETA_OVERWRITE : GEMM_BETA_ACCUMULATE;
        graph::gemm(
            grad_output->data(),
            weight_nn->data(),
            grad_input->data(),
            GEMM_ALPHA,
            beta,
            NO_TRANSPOSE,
            TRANSPOSE,
            GEMM_NDIM_MATRIX,
            NO_BATCH_DIM);
    }
}

std::string LinearManual::repr() const
{
    std::string result = "LinearManual(in=" + std::to_string(input_dim_) +
                         ", out=" + std::to_string(output_dim_);
    if(bias_tensor_ != nullptr)
    {
        result += ", bias=true";
    }
    result += ")";
    return result;
}

} // namespace nntile::module
