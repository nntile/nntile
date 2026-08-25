/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_log_softmax.cpp
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

void check_log_softmax_input(
    const at::Tensor &self,
    int64_t dim,
    bool half_to_float,
    const std::optional<at::Tensor> &out = std::nullopt)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "nntile log_softmax: expected nntile");
    if (out.has_value())
    {
        TORCH_CHECK(
            is_nntile_device(out->device()),
            "nntile log_softmax.out: expected nntile output");
    }
    TORCH_CHECK(
        !half_to_float,
        "nntile log_softmax does not support half_to_float=True");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile log_softmax supports float32 only");
    TORCH_CHECK(
        self.dim() > 0,
        "nntile log_softmax: cannot compute on empty tensor");
    at::maybe_wrap_dim(dim, self.dim());
    if (out.has_value())
    {
        TORCH_CHECK(
            out->sizes() == self.sizes(),
            "nntile log_softmax.out: output shape mismatch");
    }
}

void run_log_softmax(
    const at::Tensor &self,
    int64_t dim,
    at::Tensor &out)
{
    const int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
    tensor_log_softmax_fp32(self, out, wrapped_dim);
}

void check_log_softmax_backward(
    const at::Tensor &grad_output,
    const at::Tensor &output,
    int64_t dim,
    at::ScalarType input_dtype)
{
    TORCH_CHECK(
        is_nntile_device(grad_output.device()) &&
            is_nntile_device(output.device()),
        "nntile log_softmax_backward expects nntile tensors");
    TORCH_CHECK(
        input_dtype == at::ScalarType::Float,
        "nntile log_softmax_backward supports float32 input only");
    TORCH_CHECK(
        grad_output.scalar_type() == at::ScalarType::Float &&
            output.scalar_type() == at::ScalarType::Float,
        "nntile log_softmax_backward supports float32 only");
    TORCH_CHECK(
        grad_output.sizes() == output.sizes(),
        "nntile log_softmax_backward: shape mismatch");
    TORCH_CHECK(
        output.dim() > 0,
        "nntile log_softmax_backward: cannot compute on empty tensor");
    at::maybe_wrap_dim(dim, output.dim());
}

} // namespace

at::Tensor log_softmax(
    const at::Tensor &self,
    int64_t dim,
    bool half_to_float)
{
    nntile::GraphFillScope record;
    check_log_softmax_input(self, dim, half_to_float);
    at::Tensor out = at::empty_like(self);
    run_log_softmax(self, dim, out);
    return out;
}

at::Tensor &log_softmax_out(
    const at::Tensor &self,
    int64_t dim,
    bool half_to_float,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    check_log_softmax_input(self, dim, half_to_float, out);
    run_log_softmax(self, dim, out);
    return out;
}

at::Tensor log_softmax_backward_data(
    const at::Tensor &grad_output,
    const at::Tensor &output,
    int64_t dim,
    at::ScalarType input_dtype)
{
    nntile::GraphFillScope record;
    check_log_softmax_backward(grad_output, output, dim, input_dtype);
    at::Tensor grad_input = at::empty_like(output);
    const int64_t wrapped_dim = at::maybe_wrap_dim(dim, output.dim());
    tensor_log_softmax_backward_fp32(
        grad_output,
        output,
        grad_input,
        wrapped_dim);
    return grad_input;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("_log_softmax", TORCH_FN(torch_nntile::log_softmax));
    m.impl("_log_softmax.out", TORCH_FN(torch_nntile::log_softmax_out));
    m.impl(
        "_log_softmax_backward_data",
        TORCH_FN(torch_nntile::log_softmax_backward_data));
}
