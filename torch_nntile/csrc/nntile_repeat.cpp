/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_repeat.cpp
 * ``aten::repeat`` for device nntile via chained ``scale_slice``.
 */

#include "nntile_broadcast.h"
#include "nntile_graph_recorder_impl.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <vector>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

std::vector<int64_t> repeat_output_shape(
    c10::IntArrayRef input_shape,
    c10::IntArrayRef repeats)
{
    const std::size_t input_ndim = input_shape.size();
    const std::size_t repeat_ndim = repeats.size();
    TORCH_CHECK(
        repeat_ndim >= input_ndim,
        "Number of dimensions of repeat dims can not be smaller than number "
        "of dimensions of tensor");

    const std::size_t pad = repeat_ndim - input_ndim;
    std::vector<int64_t> out_shape;
    out_shape.reserve(repeat_ndim);
    for (std::size_t i = 0; i < repeat_ndim; ++i)
    {
        const int64_t eff =
            (i < pad) ? 1 : input_shape[static_cast<std::size_t>(i - pad)];
        TORCH_CHECK(
            repeats[static_cast<std::int64_t>(i)] >= 0,
            "repeats can not be negative");
        out_shape.push_back(eff * repeats[static_cast<std::int64_t>(i)]);
    }
    return out_shape;
}

void check_repeat_input(const at::Tensor &self)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "nntile repeat: expected nntile tensor");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile repeat supports float32 only");
    TORCH_CHECK(self.is_contiguous(), "nntile repeat requires contiguous input");
}

void run_repeat(const at::Tensor &self, at::Tensor &out, c10::IntArrayRef repeats)
{
    pin_graph_op_inputs({self});
    pin_graph_op_output(out, true);
    tensor_repeat_fp32(
        self.data_ptr<float>(),
        out.data_ptr<float>(),
        self.sizes(),
        repeats,
        out.sizes());
}

} // namespace

at::Tensor repeat_tensor(const at::Tensor &self, c10::IntArrayRef repeats)
{
    check_repeat_input(self);
    const std::vector<int64_t> out_shape =
        repeat_output_shape(self.sizes(), repeats);
    at::Tensor out = at::empty(
        out_shape,
        self.options().memory_format(at::MemoryFormat::Contiguous));
    run_repeat(self, out, repeats);
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("repeat", TORCH_FN(torch_nntile::repeat_tensor));
}
