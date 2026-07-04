/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_transpose.cpp
 */

#include "nntile_transpose.h"

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"

#include <ATen/TensorUtils.h>

#include <vector>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_model_transpose_input(
    const at::Tensor &tensor,
    int64_t model_ndim,
    const char *name)
{
    TORCH_CHECK(
        is_nntile_device(tensor.device()),
        "nntile model_transpose: expected nntile ",
        name);
    TORCH_CHECK(
        tensor.scalar_type() == at::ScalarType::Float,
        "nntile model_transpose supports float32 only");
    TORCH_CHECK(tensor.is_contiguous(), "nntile model_transpose requires contiguous");
    const int64_t n = tensor.dim();
    TORCH_CHECK(
        model_ndim > 0 && model_ndim < n,
        "nntile model_transpose: model_ndim must be in (0, ",
        n,
        "), got ",
        model_ndim);
}

std::vector<int64_t> permuted_sizes(
    c10::IntArrayRef sizes,
    int64_t rot)
{
    const int64_t n = static_cast<int64_t>(sizes.size());
    std::vector<int64_t> out(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i)
    {
        out[static_cast<std::size_t>(i)] = sizes[(i + rot) % n];
    }
    return out;
}

} // namespace

at::Tensor model_transpose_forward(
    const at::Tensor &x,
    int64_t model_ndim)
{
    check_model_transpose_input(x, model_ndim, "input");
    const int64_t n = x.dim();
    const int64_t tensor_ndim = n - model_ndim;
    at::Tensor out = at::empty(
        permuted_sizes(x.sizes(), tensor_ndim),
        x.options().memory_format(at::MemoryFormat::Contiguous));
    pin_graph_op_inputs({x});
    pin_graph_op_output(out, true);
    tensor_model_transpose_forward_fp32(
        x.data_ptr<float>(),
        x.sizes(),
        out.data_ptr<float>(),
        model_ndim);
    return out;
}

at::Tensor model_transpose_backward(
    const at::Tensor &grad_out,
    int64_t model_ndim)
{
    check_model_transpose_input(grad_out, model_ndim, "grad_out");
    at::Tensor grad_x = at::empty(
        permuted_sizes(grad_out.sizes(), model_ndim),
        grad_out.options().memory_format(at::MemoryFormat::Contiguous));
    pin_graph_op_inputs({grad_out});
    pin_graph_op_output(grad_x, false);
    tensor_model_transpose_backward_fp32(
        grad_out.data_ptr<float>(),
        grad_out.sizes(),
        grad_x.data_ptr<float>(),
        model_ndim);
    return grad_x;
}

} // namespace torch_nntile
