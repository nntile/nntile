/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_narrow.cpp
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <c10/util/irange.h>
#include <torch/library.h>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_narrow_input(const at::Tensor &self)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "nntile narrow expects tensor on device nntile");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile narrow supports float32 only");
    TORCH_CHECK(
        self.is_contiguous(),
        "nntile narrow requires contiguous tensor");
    TORCH_CHECK(self.dim() > 0, "nntile narrow: cannot narrow a 0-dim tensor");
}

std::vector<int64_t> make_narrow_output_shape(
    c10::IntArrayRef input_shape,
    int64_t dim,
    int64_t length)
{
    std::vector<int64_t> out_shape = input_shape.vec();
    out_shape[static_cast<std::size_t>(dim)] = length;
    return out_shape;
}

void run_narrow(
    const at::Tensor &self,
    int64_t dim,
    int64_t start,
    int64_t length,
    at::Tensor &out)
{
    pin_graph_op_inputs({self});
    pin_graph_op_output(out, true);
    tensor_narrow_fp32(self, dim, start, length, out);
}

} // namespace

at::Tensor narrow(
    const at::Tensor &self,
    int64_t dim,
    c10::SymInt start,
    c10::SymInt length)
{
    check_narrow_input(self);
    const int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
    const int64_t start_val = start.expect_int();
    const int64_t length_val = length.expect_int();
    const int64_t dim_size = self.size(wrapped_dim);

    TORCH_CHECK(
        start_val >= 0 && start_val <= dim_size,
        "nntile narrow: start out of range");
    TORCH_CHECK(
        length_val >= 0 && start_val + length_val <= dim_size,
        "nntile narrow: length out of range");

    at::Tensor out = at::empty(
        make_narrow_output_shape(self.sizes(), wrapped_dim, length_val),
        self.options().memory_format(at::MemoryFormat::Contiguous));
    run_narrow(self, wrapped_dim, start_val, length_val, out);
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("narrow", TORCH_FN(torch_nntile::narrow));
}
