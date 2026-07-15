/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_sum.cpp
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

#include <algorithm>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_sum_input(const at::Tensor &self)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "nntile sum expects tensor on device nntile");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile sum supports float32 only");
    TORCH_CHECK(
        self.is_contiguous(),
        "nntile sum requires contiguous tensor");
    TORCH_CHECK(self.dim() > 0, "nntile sum: cannot sum a 0-dim tensor");
}

std::vector<int64_t> infer_sum_output_sizes(
    c10::IntArrayRef input_sizes,
    at::OptionalIntArrayRef dim,
    bool keepdim)
{
    const int64_t rank = static_cast<int64_t>(input_sizes.size());
    std::vector<int64_t> reduce_dims;
    if (!dim.has_value() || dim->empty())
    {
        reduce_dims.reserve(static_cast<std::size_t>(rank));
        for (int64_t i = 0; i < rank; ++i)
        {
            reduce_dims.push_back(i);
        }
    }
    else
    {
        reduce_dims.reserve(dim->size());
        for (const auto d : *dim)
        {
            const int64_t axis = d < 0 ? d + rank : d;
            TORCH_CHECK(
                axis >= 0 && axis < rank,
                "nntile sum: dimension out of range");
            reduce_dims.push_back(axis);
        }
    }

    std::vector<int64_t> out_sizes;
    out_sizes.reserve(static_cast<std::size_t>(rank));
    for (int64_t i = 0; i < rank; ++i)
    {
        const bool reduce = std::find(
                                reduce_dims.begin(),
                                reduce_dims.end(),
                                i) != reduce_dims.end();
        if (reduce)
        {
            if (keepdim)
            {
                out_sizes.push_back(1);
            }
        }
        else
        {
            out_sizes.push_back(input_sizes[static_cast<std::size_t>(i)]);
        }
    }
    return out_sizes;
}

} // namespace

at::Tensor sum_dimlist(
    const at::Tensor &self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> /*dtype*/)
{
    check_sum_input(self);
    const std::vector<int64_t> out_sizes =
        infer_sum_output_sizes(self.sizes(), dim, keepdim);
    at::Tensor out = at::empty(
        out_sizes,
        self.options().memory_format(at::MemoryFormat::Contiguous));
    tensor_sum_dimlist_fp32(self, out, dim, keepdim);
    return out;
}

at::Tensor &sum_dimlist_out(
    const at::Tensor &self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype,
    at::Tensor &out)
{
    check_sum_input(self);
    TORCH_CHECK(
        is_nntile_device(out.device()),
        "nntile sum.out expects output on device nntile");
    const std::vector<int64_t> out_sizes =
        infer_sum_output_sizes(self.sizes(), dim, keepdim);
    TORCH_CHECK(
        out.sizes().vec() == out_sizes,
        "nntile sum.out: output shape mismatch");
    TORCH_CHECK(out.is_contiguous(), "nntile sum.out requires contiguous out");
    tensor_sum_dimlist_fp32(self, out, dim, keepdim);
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("sum.IntList_out", TORCH_FN(torch_nntile::sum_dimlist_out));
    m.impl("sum.dim_IntList", TORCH_FN(torch_nntile::sum_dimlist));
}
