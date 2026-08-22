/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_norm.cpp
 */

#include "nntile_norm.h"

#include "nntile_executor.h"

#include "nntile_broadcast.h"
#include "nntile_graph_recorder_impl.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <cmath>
#include <vector>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_norm_input(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(
        is_nntile_device(tensor.device()),
        "nntile norm: expected nntile ",
        name);
    TORCH_CHECK(
        tensor.scalar_type() == at::ScalarType::Float,
        "nntile norm supports float32 only");
    TORCH_CHECK(tensor.is_contiguous(), "nntile norm requires contiguous");
}

bool is_two_norm(const at::Scalar &ord)
{
    if (!ord.isFloatingPoint())
    {
        return ord.to<int64_t>() == 2;
    }
    return std::fabs(ord.to<double>() - 2.0) < 1e-6;
}

int64_t normalize_dim(int64_t dim, int64_t ndim)
{
    if (dim < 0)
    {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "nntile norm: dim out of range");
    return dim;
}

std::vector<int64_t> reduced_sizes(
    c10::IntArrayRef input_shape,
    int64_t axis)
{
    auto sizes = input_shape.vec();
    sizes.erase(sizes.begin() + static_cast<std::size_t>(axis));
    return sizes;
}

} // namespace

std::tuple<at::Tensor, at::Tensor> norm_forward(
    const at::Tensor &input,
    std::optional<int64_t> dim,
    bool keepdim,
    at::Tensor *out)
{
    check_norm_input(input, "input");
    const at::Tensor &x = input;

    if (!dim.has_value())
    {
        const int64_t numel = x.numel();
        TORCH_CHECK(numel > 0, "nntile norm: cannot compute norm of empty tensor");
        at::Tensor x_flat = x.view({numel});
        at::Tensor norm_values = at::empty({}, x.options());

        tensor_norm_fp32(x_flat, norm_values);

        if (keepdim)
        {
            std::vector<int64_t> sizes(
                static_cast<std::size_t>(x.dim()),
                1);
            at::Tensor output;
            if (out != nullptr)
            {
                TORCH_CHECK(
                    out->sizes() == c10::IntArrayRef(sizes),
                    "nntile norm: output tensor shape mismatch");
                TORCH_CHECK(
                    out->is_contiguous(),
                    "nntile norm: output tensor must be contiguous");
                output = *out;
            }
            else
            {
                output = at::empty(sizes, x.options());
            }
            tensor_broadcast_scalar_fp32(norm_values, output);
            return {output, norm_values};
        }

        if (out != nullptr)
        {
            TORCH_CHECK(
                out->sizes().empty(),
                "nntile norm: output tensor shape mismatch");
            TORCH_CHECK(
                out->is_contiguous(),
                "nntile norm: output tensor must be contiguous");
            nntile::TensorGraph::TensorNode *node = lookup_data_node(
                norm_values,
                {});
            TORCH_CHECK(
                node != nullptr,
                "nntile norm: scalar norm output node is missing");
            register_data_node(*out, node);
            return {*out, norm_values};
        }

        return {norm_values, norm_values};
    }

    const int64_t axis = normalize_dim(*dim, x.dim());
    const std::vector<int64_t> reduced_sizes_vec =
        reduced_sizes(x.sizes(), axis);
    at::IntArrayRef reduced_sizes_ref(reduced_sizes_vec);

    at::Tensor output;
    if (out != nullptr)
    {
        if (keepdim)
        {
            auto sizes = x.sizes().vec();
            sizes[static_cast<std::size_t>(axis)] = 1;
            TORCH_CHECK(
                out->sizes() == c10::IntArrayRef(sizes),
                "nntile norm: output tensor shape mismatch");
        }
        else
        {
            TORCH_CHECK(
                out->sizes() == reduced_sizes_ref,
                "nntile norm: output tensor shape mismatch");
        }
        TORCH_CHECK(
            out->is_contiguous(),
            "nntile norm: output tensor must be contiguous");
        output = *out;
    }
    else if (keepdim)
    {
        auto sizes = x.sizes().vec();
        sizes[static_cast<std::size_t>(axis)] = 1;
        output = at::empty(sizes, x.options());
    }
    else
    {
        output = at::empty(reduced_sizes_ref, x.options());
    }
    at::Tensor norm_values = at::empty(reduced_sizes_ref, x.options());

    tensor_norm_slice_fp32(x, output, axis, keepdim);
    tensor_norm_slice_fp32(x, norm_values, axis, false);
    return {output, norm_values};
}

at::Tensor linalg_vector_norm_nntile(
    const at::Tensor &self,
    const at::Scalar &ord,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype)
{
    nntile::GraphFillScope record;
    if (!is_nntile_device(self.device()))
    {
        return at::linalg_vector_norm(self, ord, dim, keepdim, dtype);
    }
    TORCH_CHECK(
        is_two_norm(ord),
        "nntile linalg_vector_norm supports ord=2 only");
    TORCH_CHECK(
        !dtype.has_value(),
        "nntile linalg_vector_norm does not support dtype conversion");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile linalg_vector_norm supports float32 only");
    TORCH_CHECK(
        !self.requires_grad() || !at::GradMode::is_enabled(),
        "nntile linalg_vector_norm is forward-only; call it under "
        "torch.no_grad() or detach the input");

  std::optional<int64_t> axis;
  if (dim.has_value())
  {
    TORCH_CHECK(
        dim->size() == 1,
        "nntile linalg_vector_norm supports a single dim only");
    axis = (*dim)[0];
  }

  auto [output, norm_values] = norm_forward(self, axis, keepdim, nullptr);
  (void)norm_values;
  return output;
}

at::Tensor &linalg_vector_norm_out_nntile(
    const at::Tensor &self,
    const at::Scalar &ord,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    if (!is_nntile_device(self.device()))
    {
        out.copy_(at::linalg_vector_norm(self, ord, dim, keepdim, dtype));
        return out;
    }
    TORCH_CHECK(
        is_nntile_device(out.device()),
        "nntile linalg_vector_norm.out: output must be on nntile");
    TORCH_CHECK(
        is_two_norm(ord),
        "nntile linalg_vector_norm supports ord=2 only");
    TORCH_CHECK(
        !dtype.has_value(),
        "nntile linalg_vector_norm does not support dtype conversion");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile linalg_vector_norm supports float32 only");
    TORCH_CHECK(
        !self.requires_grad() || !at::GradMode::is_enabled(),
        "nntile linalg_vector_norm is forward-only; call it under "
        "torch.no_grad() or detach the input");

    std::optional<int64_t> axis;
    if (dim.has_value())
    {
        TORCH_CHECK(
            dim->size() == 1,
            "nntile linalg_vector_norm supports a single dim only");
        axis = (*dim)[0];
    }

    auto [output, norm_values] = norm_forward(self, axis, keepdim, &out);
    (void)norm_values;
    (void)output;
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(
        "linalg_vector_norm",
        TORCH_FN(torch_nntile::linalg_vector_norm_nntile));
    m.impl(
        "linalg_vector_norm.out",
        TORCH_FN(torch_nntile::linalg_vector_norm_out_nntile));
}
