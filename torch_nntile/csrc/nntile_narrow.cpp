/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_narrow.cpp
 * Zero-copy ``aten::narrow`` view (sizes/strides/offset).
 */

#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_meta.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

#include <optional>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

at::Tensor make_strided_view(
    const at::Tensor &self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    int64_t storage_offset)
{
    at::Tensor result = at::detail::make_tensor<at::TensorImpl>(
        c10::Storage(self.storage()),
        self.key_set(),
        self.dtype());
    auto *result_impl = result.unsafeGetTensorImpl();
    result_impl->set_storage_offset(storage_offset);
    result_impl->set_sizes_and_strides(size, stride);
    record_view_alias(self, result);
    return result;
}

} // namespace

at::Tensor narrow(
    const at::Tensor &self,
    int64_t dim,
    c10::SymInt start,
    c10::SymInt length)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "nntile narrow expects tensor on device nntile");
    TORCH_CHECK(self.dim() > 0, "nntile narrow: cannot narrow a 0-dim tensor");
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

    auto sizes = self.sizes().vec();
    sizes[static_cast<std::size_t>(wrapped_dim)] = length_val;
    const int64_t offset =
        self.storage_offset() +
        start_val * self.stride(wrapped_dim);
    return make_strided_view(
        self,
        sizes,
        self.strides(),
        offset);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("narrow", TORCH_FN(torch_nntile::narrow));
}

} // namespace torch_nntile
