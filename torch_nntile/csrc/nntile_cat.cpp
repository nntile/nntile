/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_cat.cpp
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/IListRef.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#include <vector>

namespace torch_nntile
{

// densify_cat_inputs must call this C++ kernel — not Tensor::contiguous().
// The Tensor method returns *this when is_contiguous() without dispatching
// to PrivateUse1, so same-numel reshape views (View Backward) stay aliased
// to the parent TensorNode and SplitBackward cat packs wrong tile shapes.
at::Tensor contiguous(
    const at::Tensor &self,
    at::MemoryFormat memory_format);

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

std::vector<at::Tensor> materialize_cat_inputs(
    const at::ITensorListRef &tensors)
{
    std::vector<at::Tensor> materialized;
    materialized.reserve(tensors.size());
    for (const at::Tensor &tensor : tensors)
    {
        materialized.push_back(tensor);
    }
    TORCH_CHECK(
        !materialized.empty(),
        "torch.cat(): expected a non-empty list of Tensors");
    return materialized;
}

void check_cat_tensor(const at::Tensor &tensor)
{
    TORCH_CHECK(
        is_nntile_device(tensor.device()),
        "nntile cat expects all tensors on device nntile");
    TORCH_CHECK(
        tensor.scalar_type() == at::ScalarType::Float,
        "nntile cat supports float32 only");
}

std::vector<at::Tensor> densify_cat_inputs(
    const std::vector<at::Tensor> &tensors)
{
    std::vector<at::Tensor> out;
    out.reserve(tensors.size());
    for (const at::Tensor &tensor : tensors)
    {
        check_cat_tensor(tensor);
        // Always densify via the PrivateUse1 kernel (not Tensor::contiguous):
        // reshape views can be is_contiguous yet still share a differently
        // shaped TensorNode (View Backward of attention heads).
        out.push_back(
            contiguous(tensor, at::MemoryFormat::Contiguous));
    }
    return out;
}

void check_cat_inputs(
    const std::vector<at::Tensor> &tensors,
    int64_t dim,
    const std::optional<at::Tensor> &out = std::nullopt)
{
    for (const auto i : c10::irange(static_cast<int64_t>(tensors.size())))
    {
        TORCH_CHECK(
            tensors[static_cast<std::size_t>(i)].dim() > 0,
            "zero-dimensional tensor (at position ",
            i,
            ") cannot be concatenated");
    }
    check_cat_tensor(tensors[0]);
    dim = at::maybe_wrap_dim(dim, tensors[0].dim());

    std::vector<int64_t> out_sizes = tensors[0].sizes().vec();
    for (const auto i : c10::irange(1, static_cast<int64_t>(tensors.size())))
    {
        check_cat_tensor(tensors[static_cast<std::size_t>(i)]);
        const int64_t first_dims = tensors[0].dim();
        const int64_t second_dims =
            tensors[static_cast<std::size_t>(i)].dim();
        TORCH_CHECK(
            first_dims == second_dims,
            "Tensors must have same number of dimensions: got ",
            first_dims,
            " and ",
            second_dims);
        for (const auto axis : c10::irange(first_dims))
        {
            if (axis == dim)
            {
                continue;
            }
            const int64_t first_dim_size = tensors[0].sizes()[axis];
            const int64_t second_dim_size =
                tensors[static_cast<std::size_t>(i)].sizes()[axis];
            TORCH_CHECK(
                first_dim_size == second_dim_size,
                "Sizes of tensors must match except in dimension ",
                dim,
                ". Expected size ",
                static_cast<long long>(first_dim_size),
                " but got size ",
                static_cast<long long>(second_dim_size),
                " for tensor number ",
                i,
                " in the list.");
        }
        out_sizes[static_cast<std::size_t>(dim)] +=
            tensors[static_cast<std::size_t>(i)].size(dim);
    }

    if (out.has_value())
    {
        TORCH_CHECK(
            is_nntile_device(out->device()),
            "nntile cat.out expects output on device nntile");
        TORCH_CHECK(
            out->scalar_type() == at::ScalarType::Float,
            "nntile cat.out supports float32 only");
        TORCH_CHECK(
            out->is_contiguous(),
            "nntile cat.out requires contiguous output");
        TORCH_CHECK(
            out->sizes() == c10::IntArrayRef(out_sizes),
            "nntile cat.out: output shape mismatch");
    }
}

at::Tensor make_cat_output(
    const std::vector<at::Tensor> &tensors,
    int64_t dim)
{
    std::vector<int64_t> out_sizes = tensors[0].sizes().vec();
    for (const auto i : c10::irange(1, static_cast<int64_t>(tensors.size())))
    {
        out_sizes[static_cast<std::size_t>(dim)] +=
            tensors[static_cast<std::size_t>(i)].size(dim);
    }
    return at::empty(
        out_sizes,
        tensors[0].options().memory_format(at::MemoryFormat::Contiguous));
}

void run_cat(
    const std::vector<at::Tensor> &tensors,
    int64_t dim,
    at::Tensor &out)
{
    tensor_cat_fp32(tensors, out, dim);
}

} // namespace

at::Tensor cat(const at::ITensorListRef &tensors, int64_t dim)
{
    std::vector<at::Tensor> materialized =
        densify_cat_inputs(materialize_cat_inputs(tensors));
    if (materialized.size() == 1)
    {
        return materialized[0];
    }

    const int64_t wrapped_dim =
        at::maybe_wrap_dim(dim, materialized[0].dim());
    check_cat_inputs(materialized, wrapped_dim);
    at::Tensor out = make_cat_output(materialized, wrapped_dim);
    run_cat(materialized, wrapped_dim, out);
    return out;
}

at::Tensor &cat_out(
    const at::ITensorListRef &tensors,
    int64_t dim,
    at::Tensor &out)
{
    std::vector<at::Tensor> materialized =
        densify_cat_inputs(materialize_cat_inputs(tensors));
    if (materialized.size() == 1)
    {
        TORCH_CHECK(
            out.sizes() == materialized[0].sizes(),
            "nntile cat.out: output shape mismatch for single input");
        out.copy_(materialized[0]);
        return out;
    }

    const int64_t wrapped_dim =
        at::maybe_wrap_dim(dim, materialized[0].dim());
    check_cat_inputs(materialized, wrapped_dim, out);
    run_cat(materialized, wrapped_dim, out);
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("cat", TORCH_FN(torch_nntile::cat));
    m.impl("cat.out", TORCH_FN(torch_nntile::cat_out));
}
