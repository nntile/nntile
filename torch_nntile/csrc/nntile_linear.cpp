/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_linear.cpp
 */

#include "nntile_executor.h"
#include "nntile_gemm_layout.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_layout_checks.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>

#include <array>
#include <chrono>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

std::vector<nntile::Index> pytorch_shape_to_graph(c10::IntArrayRef shape)
{
    std::vector<nntile::Index> graph_shape;
    graph_shape.reserve(shape.size());
    for (const auto dim : shape)
    {
        graph_shape.push_back(static_cast<nntile::Index>(dim));
    }
    return graph_shape;
}

void check_linear_bias(
    const at::Tensor &bias,
    const at::Tensor &weight)
{
    TORCH_CHECK(
        is_nntile_device(bias.device()),
        "nntile linear: bias must be on device nntile");
    TORCH_CHECK(
        bias.scalar_type() == at::ScalarType::Float,
        "nntile linear: bias must be float32");
    TORCH_CHECK(
        bias.is_contiguous(),
        "nntile linear: bias must be contiguous");
    TORCH_CHECK(bias.dim() == 1, "nntile linear: bias must be 1D");
    TORCH_CHECK(
        bias.size(0) == weight.size(0),
        "nntile linear: bias size must equal out_features");
}

void check_linear_tensors(
    const at::Tensor &input,
    const at::Tensor &weight,
    const std::optional<at::Tensor> &bias,
    const std::optional<at::Tensor> &out = std::nullopt)
{
    TORCH_CHECK(
        is_nntile_device(input.device()) &&
            is_nntile_device(weight.device()),
        "nntile linear expects input and weight on device nntile");
    if (out.has_value())
    {
        TORCH_CHECK(
            is_nntile_device(out->device()),
            "nntile linear.out expects output on device nntile");
    }
    TORCH_CHECK(input.dim() >= 1, "nntile linear: input must be at least 1D");
    TORCH_CHECK(weight.dim() == 2, "nntile linear: weight must be 2D");
    TORCH_CHECK(
        input.size(-1) == weight.size(1),
        "nntile linear: feature dimension mismatch");
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::Float &&
            weight.scalar_type() == at::ScalarType::Float,
        "nntile linear supports float32 only");
    if (bias.has_value() && bias->defined())
    {
        check_linear_bias(*bias, weight);
    }
}

at::Tensor make_linear_output(
    const std::vector<int64_t> &out_shape,
    const at::Tensor &input)
{
    std::vector<int64_t> sizes(out_shape.begin(), out_shape.end());
    return at::empty(
        sizes,
        input.options().memory_format(at::MemoryFormat::Contiguous));
}

void run_linear(
    const PreparedGemmOperands &prepared,
    at::Tensor &output,
    const std::optional<at::Tensor> &bias)
{
    const bool has_bias = bias.has_value() && bias->defined();
    if (has_bias)
    {
        tensor_linear_bias_fp32(
            prepared.a,
            prepared.b,
            *bias,
            output);
    }
    else
    {
        tensor_linear_fp32(prepared.a, prepared.b, output);
    }
}

GemmMatrixLayout linear_operand_layout(const at::Tensor &tensor)
{
    if (tensor.dim() == 1)
    {
        GemmMatrixLayout layout;
        layout.gemm_shape = {1, tensor.size(0)};
        layout.trans = false;
        layout.needs_copy = !tensor.is_contiguous();
        return layout;
    }
    GemmMatrixLayout layout;
    layout.gemm_shape = pytorch_sizes_vector(tensor.sizes());
    layout.trans = false;
    layout.needs_copy = !tensor.is_contiguous();
    return layout;
}

} // namespace

at::Tensor linear(
    const at::Tensor &input,
    const at::Tensor &weight,
    const std::optional<at::Tensor> &bias)
{
    nntile::GraphFillScope record;
    require_nntile_kernel_dense(input, "linear input");
    require_nntile_kernel_dense(weight, "linear weight");
    check_linear_tensors(input, weight, bias);
    const PreparedGemmOperands prepared =
        prepare_linear_operands(input, weight);
    at::Tensor output = make_linear_output(prepared.out_shape, input);
    run_linear(prepared, output, bias);
    return output;
}

at::Tensor &linear_out(
    const at::Tensor &input,
    const at::Tensor &weight,
    const std::optional<at::Tensor> &bias,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    require_nntile_kernel_dense(input, "linear input");
    require_nntile_kernel_dense(weight, "linear weight");
    check_linear_tensors(input, weight, bias, out);
    const PreparedGemmOperands prepared =
        prepare_linear_operands(input, weight);
    TORCH_CHECK(
        out.sizes().vec() == prepared.out_shape,
        "nntile linear.out: output shape mismatch");
    TORCH_CHECK(
        out.is_contiguous(),
        "nntile linear.out requires contiguous out");
    run_linear(prepared, out, bias);
    return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_backward(
    const at::Tensor &input,
    const at::Tensor &grad_output,
    const at::Tensor &weight,
    std::array<bool, 3> output_mask)
{
    nntile::GraphFillScope record;
    const auto t0 = std::chrono::steady_clock::now();
    TORCH_CHECK(
        is_nntile_device(input.device()) &&
            is_nntile_device(grad_output.device()) &&
            is_nntile_device(weight.device()),
        "nntile linear_backward expects nntile tensors");
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::Float &&
            grad_output.scalar_type() == at::ScalarType::Float &&
            weight.scalar_type() == at::ScalarType::Float,
        "nntile linear_backward supports float32 only");

    const at::Tensor grad_out = grad_output;
    require_nntile_kernel_dense(grad_out, "linear_backward grad_output");
    require_nntile_kernel_dense(input, "linear_backward input");
    require_nntile_kernel_dense(weight, "linear_backward weight");

    const PreparedGemmOperands forward =
        prepare_linear_operands(input, weight);
    const GemmMatrixLayout weight_layout =
        analyze_matrix_layout_for_nntile(weight);

    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;
    if (output_mask[0])
    {
        const GemmParams grad_input_params =
            infer_linear_backward_grad_input_params(forward.params);

        const GemmMatrixLayout grad_out_layout =
            linear_operand_layout(grad_out);
        TORCH_CHECK(
            !grad_out_layout.needs_copy,
            "nntile linear_backward: grad_output must be contiguous or "
            "row/column-contiguous");
        TORCH_CHECK(
            !weight_layout.needs_copy,
            "nntile linear_backward: weight must be contiguous or "
            "row/column-contiguous");
        const at::Tensor &grad_out_prepared = grad_out;
        const at::Tensor &weight_prepared = forward.b;

        grad_input = at::empty_like(input);
        tensor_gemm_fp32(
            grad_input_params,
            grad_out_prepared,
            grad_out_layout.gemm_shape,
            weight_prepared,
            forward.b_gemm_shape,
            grad_input,
            forward.a_gemm_shape);
        nntile::TensorGraph::TensorNode *grad_input_node = lookup_data_node(
            grad_input,
            pytorch_shape_to_graph(grad_input.sizes()));
        if (grad_input_node != nullptr)
        {
            register_param_grad_node(input, grad_input_node);
            at::Tensor grad_input_alias = grad_input;
            register_grad_alias_for_host_copy(
                grad_input_alias,
                grad_input_node);
        }
    }
    if (output_mask[1])
    {
        const GemmMatrixLayout grad_out_layout =
            linear_operand_layout(grad_out);
        TORCH_CHECK(
            !grad_out_layout.needs_copy,
            "nntile linear_backward: grad_output must be contiguous or "
            "row/column-contiguous");
        TORCH_CHECK(
            forward.a.is_contiguous(),
            "nntile linear_backward: input must be contiguous");
        const at::Tensor &grad_out_prepared = grad_out;
        const at::Tensor &input_prepared = forward.a;

        grad_weight = at::empty_like(weight);
        if (weight_layout.trans)
        {
            GemmParams grad_weight_params =
                infer_linear_backward_grad_weight_params(forward.params);
            tensor_gemm_fp32(
                grad_weight_params,
                input_prepared,
                forward.a_gemm_shape,
                grad_out_prepared,
                grad_out_layout.gemm_shape,
                grad_weight,
                forward.b_gemm_shape);
        }
        else
        {
            GemmParams grad_weight_params =
                infer_linear_backward_grad_weight_params(forward.params);
            tensor_gemm_fp32(
                grad_weight_params,
                grad_out_prepared,
                grad_out_layout.gemm_shape,
                input_prepared,
                forward.a_gemm_shape,
                grad_weight,
                forward.b_gemm_shape);
        }
        nntile::TensorGraph::TensorNode *grad_node = lookup_data_node(
            grad_weight,
            pytorch_shape_to_graph(grad_weight.sizes()));
        if (grad_node != nullptr)
        {
            register_param_grad_node(weight, grad_node);
            at::Tensor grad_weight_alias = grad_weight;
            register_grad_alias_for_host_copy(grad_weight_alias, grad_node);
        }
    }
    if (output_mask[2])
    {
        TORCH_CHECK(
            grad_out.dim() >= 1,
            "nntile linear_backward: grad_output must be at least 1D");
        grad_bias = at::empty(
            {weight.size(0)},
            grad_out.options().memory_format(
                at::MemoryFormat::Contiguous));
        tensor_linear_grad_bias_fp32(grad_out, grad_bias);
        nntile::TensorGraph::TensorNode *grad_bias_node = lookup_data_node(
            grad_bias,
            pytorch_shape_to_graph(grad_bias.sizes()));
        if (grad_bias_node != nullptr)
        {
            at::Tensor grad_bias_alias = grad_bias;
            register_grad_alias_for_host_copy(
                grad_bias_alias,
                grad_bias_node);
        }
    }
    note_record_linear_bwd(
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0)
            .count());
    return {grad_input, grad_weight, grad_bias};
}

} // namespace torch_nntile

// Match device=cuda: do NOT register PrivateUse1 ``linear``. CUDA uses
// CompositeImplicitAutograd ``linear`` → ``addmm``; Autograd differentiates
// the decomposed graph (AddmmBackward). Our PrivateUse1 ``addmm`` /
// ``mm`` supply the device kernels. Keep the helpers above for direct
// TensorGraph / StarPU use.

