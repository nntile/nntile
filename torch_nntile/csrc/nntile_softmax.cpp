/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_softmax.cpp
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

void check_softmax_input(
    const at::Tensor &self,
    int64_t dim,
    bool half_to_float,
    const std::optional<at::Tensor> &out = std::nullopt)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "nntile softmax: expected nntile");
    if (out.has_value())
    {
        TORCH_CHECK(
            is_nntile_device(out->device()),
            "nntile softmax.out: expected nntile output");
    }
    TORCH_CHECK(
        !half_to_float,
        "nntile softmax does not support half_to_float=True");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile softmax supports float32 only");
    TORCH_CHECK(self.is_contiguous(), "nntile softmax requires contiguous input");
    TORCH_CHECK(
        self.dim() > 0,
        "nntile softmax: cannot compute softmax on empty tensor");
    at::maybe_wrap_dim(dim, self.dim());
    if (out.has_value())
    {
        TORCH_CHECK(
            out->sizes() == self.sizes(),
            "nntile softmax.out: output shape mismatch");
        TORCH_CHECK(
            out->is_contiguous(),
            "nntile softmax.out requires contiguous out");
    }
}

void run_softmax(
    const at::Tensor &self,
    int64_t dim,
    at::Tensor &out)
{
    const int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
    pin_graph_op_inputs({self});
    pin_graph_op_output(out, true);
    tensor_softmax_fp32(self, out, wrapped_dim);
}

} // namespace

at::Tensor softmax(
    const at::Tensor &self,
    int64_t dim,
    bool half_to_float)
{
    check_softmax_input(self, dim, half_to_float);
    at::Tensor out = at::empty_like(self);
    run_softmax(self, dim, out);
    return out;
}

at::Tensor &softmax_out(
    const at::Tensor &self,
    int64_t dim,
    bool half_to_float,
    at::Tensor &out)
{
    check_softmax_input(self, dim, half_to_float, out);
    run_softmax(self, dim, out);
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("_softmax", TORCH_FN(torch_nntile::softmax));
    m.impl("_softmax.out", TORCH_FN(torch_nntile::softmax_out));
}
