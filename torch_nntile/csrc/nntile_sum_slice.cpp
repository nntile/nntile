/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_sum_slice.cpp
 * ``sum_slice`` forward / ``add_slice`` backward for GAP.
 */

#include "nntile_sum_slice.h"

#include "nntile_executor.h"

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

void check_sum_slice_input(const at::Tensor &src, int64_t axis)
{
    TORCH_CHECK(
        is_nntile_device(src.device()),
        "nntile sum_slice expects tensor on device nntile");
    TORCH_CHECK(
        src.scalar_type() == at::ScalarType::Float,
        "nntile sum_slice supports float32 only");
    TORCH_CHECK(
        src.is_contiguous(),
        "nntile sum_slice requires contiguous tensor");
    TORCH_CHECK(
        axis >= 0 && axis < src.dim(),
        "nntile sum_slice: axis out of range");
}

std::vector<int64_t> reduced_sizes(
    c10::IntArrayRef sizes,
    int64_t axis)
{
    std::vector<int64_t> out;
    out.reserve(static_cast<std::size_t>(sizes.size() - 1));
    for (int64_t i = 0; i < sizes.size(); ++i)
    {
        if (i != axis)
        {
            out.push_back(sizes[static_cast<std::size_t>(i)]);
        }
    }
    return out;
}

} // namespace

at::Tensor sum_slice_forward(
    const at::Tensor &src,
    int64_t axis,
    double alpha,
    double beta)
{
    nntile::GraphFillScope record;
    check_sum_slice_input(src, axis);
    TORCH_CHECK(
        beta == 0.0,
        "nntile sum_slice_forward currently supports beta=0 only");
    at::Tensor out = at::empty(
        reduced_sizes(src.sizes(), axis),
        src.options().memory_format(at::MemoryFormat::Contiguous));
    tensor_sum_slice_fp32(
        src,
        out,
        axis,
        static_cast<float>(alpha),
        static_cast<float>(beta));
    return out;
}

at::Tensor sum_slice_backward(
    const at::Tensor &grad_out,
    const at::Tensor &src,
    int64_t axis,
    double alpha)
{
    nntile::GraphFillScope record;
    check_sum_slice_input(src, axis);
    TORCH_CHECK(
        is_nntile_device(grad_out.device()),
        "nntile sum_slice_backward expects nntile grad_out");
    TORCH_CHECK(
        grad_out.scalar_type() == at::ScalarType::Float,
        "nntile sum_slice_backward supports float32 only");
    TORCH_CHECK(
        grad_out.is_contiguous(),
        "nntile sum_slice_backward requires contiguous grad_out");
    TORCH_CHECK(
        grad_out.sizes().vec() == reduced_sizes(src.sizes(), axis),
        "nntile sum_slice_backward: grad_out shape mismatch");

    // Old GAP backward: add_slice_inplace(alpha, dy, 0, dx, axis).
    at::Tensor zeros = at::zeros_like(src);
    at::Tensor grad_src = at::empty(
        src.sizes(),
        src.options().memory_format(at::MemoryFormat::Contiguous));
    tensor_add_slice_fp32(
        static_cast<float>(alpha),
        grad_out,
        /*beta=*/0.0f,
        zeros,
        grad_src,
        axis);
    return grad_src;
}

} // namespace torch_nntile
