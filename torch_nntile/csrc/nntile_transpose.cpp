/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_transpose.cpp
 */

#include "nntile_model_transpose.h"

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_gc.h"

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
    tensor_model_transpose_forward_fp32(x, out, model_ndim);
    return out;
}

at::Tensor model_transpose_backward(
    const at::Tensor &grad_out,
    int64_t model_ndim,
    const at::Tensor &x)
{
    check_model_transpose_input(grad_out, model_ndim, "grad_out");
    at::Tensor grad_x = empty_metadata_tensor(
        permuted_sizes(grad_out.sizes(), model_ndim),
        grad_out.scalar_type(),
        grad_out.device());
    tensor_model_transpose_backward_fp32(grad_out, grad_x, model_ndim);
#ifdef TORCH_NNTILE_USE_LIBNNTILE
    if (x.defined())
    {
        std::vector<nntile::Index> grad_shape;
        grad_shape.reserve(static_cast<std::size_t>(grad_x.dim()));
        for (const auto dim : grad_x.sizes())
        {
            grad_shape.push_back(static_cast<nntile::Index>(dim));
        }
        nntile::TensorGraph::TensorNode *grad_x_node = lookup_data_node(
            grad_x,
            grad_shape);
        if (grad_x_node != nullptr)
        {
            register_param_grad_node(x, grad_x_node);
            at::Tensor grad_x_alias = grad_x;
            register_grad_alias_for_host_copy(grad_x_alias, grad_x_node);
        }
    }
#endif
    return grad_x;
}

} // namespace torch_nntile
