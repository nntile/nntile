/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_softmax_backward.cpp
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_softmax_backward(
    const at::Tensor &grad_output,
    const at::Tensor &output,
    int64_t dim,
    at::ScalarType input_dtype)
{
    TORCH_CHECK(
        is_nntile_device(grad_output.device()) &&
            is_nntile_device(output.device()),
        "nntile softmax_backward expects nntile tensors");
    TORCH_CHECK(
        input_dtype == at::ScalarType::Float,
        "nntile softmax_backward supports float32 input only");
    TORCH_CHECK(
        grad_output.scalar_type() == at::ScalarType::Float &&
            output.scalar_type() == at::ScalarType::Float,
        "nntile softmax_backward supports float32 only");
    TORCH_CHECK(
        grad_output.sizes() == output.sizes(),
        "nntile softmax_backward: shape mismatch");
    TORCH_CHECK(
        grad_output.is_contiguous() && output.is_contiguous(),
        "nntile softmax_backward requires contiguous tensors");
    TORCH_CHECK(
        output.dim() > 0,
        "nntile softmax_backward: cannot compute on empty tensor");
    at::maybe_wrap_dim(dim, output.dim());
}

void run_softmax_backward(
    const at::Tensor &grad_output,
    const at::Tensor &output,
    int64_t dim,
    at::Tensor &grad_input)
{
    const int64_t wrapped_dim = at::maybe_wrap_dim(dim, output.dim());
    tensor_softmax_backward_fp32(
        grad_output,
        output,
        grad_input,
        wrapped_dim);
}

} // namespace

at::Tensor softmax_backward_data(
    const at::Tensor &grad_output,
    const at::Tensor &output,
    int64_t dim,
    at::ScalarType input_dtype)
{
    nntile::GraphFillScope record;
    check_softmax_backward(grad_output, output, dim, input_dtype);
    at::Tensor grad_input = at::empty_like(output);
    run_softmax_backward(grad_output, output, dim, grad_input);
    return grad_input;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(
        "_softmax_backward_data",
        TORCH_FN(torch_nntile::softmax_backward_data));
}
