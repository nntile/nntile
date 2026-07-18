/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_split.cpp
 * Zero-copy ``aten::split`` / ``chunk`` via narrow views.
 */

#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_meta.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#include <numeric>
#include <vector>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_split_input(const at::Tensor &self)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "nntile split expects tensor on device nntile");
    TORCH_CHECK(self.dim() > 0, "nntile split: cannot split a 0-dim tensor");
}

std::vector<int64_t> symint_array_to_int64(c10::SymIntArrayRef sizes)
{
    std::vector<int64_t> out;
    out.reserve(sizes.size());
    for (const c10::SymInt &size : sizes)
    {
        out.push_back(size.expect_int());
    }
    return out;
}

void validate_split_sizes(
    int64_t dim_size,
    const std::vector<int64_t> &split_sizes)
{
    TORCH_CHECK(
        !split_sizes.empty(),
        "nntile split: split_sizes must be non-empty");
    int64_t total = 0;
    for (const auto i : c10::irange(split_sizes.size()))
    {
        TORCH_CHECK(
            split_sizes[static_cast<std::size_t>(i)] >= 0,
            "nntile split: split_sizes must be non-negative");
        total += split_sizes[static_cast<std::size_t>(i)];
    }
    TORCH_CHECK(
        total == dim_size,
        "nntile split: split_sizes sum to ",
        static_cast<long long>(total),
        " but dim size is ",
        static_cast<long long>(dim_size));
}

std::vector<int64_t> compute_equal_split_sizes(
    int64_t dim_size,
    int64_t split_size)
{
    TORCH_CHECK(split_size > 0, "nntile split: split_size must be positive");
    std::vector<int64_t> sizes;
    for (int64_t offset = 0; offset < dim_size; offset += split_size)
    {
        sizes.push_back(std::min(split_size, dim_size - offset));
    }
    TORCH_CHECK(!sizes.empty(), "nntile split: empty result");
    return sizes;
}

std::vector<int64_t> compute_chunk_sizes(int64_t dim_size, int64_t chunks)
{
    TORCH_CHECK(chunks > 0, "nntile chunk: chunks must be positive");
    const int64_t split_size = (dim_size + chunks - 1) / chunks;
    return compute_equal_split_sizes(dim_size, split_size);
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

at::Tensor narrow_view(
    const at::Tensor &self,
    int64_t dim,
    int64_t start,
    int64_t length)
{
    auto sizes = self.sizes().vec();
    sizes[static_cast<std::size_t>(dim)] = length;
    const int64_t offset =
        self.storage_offset() + start * self.stride(dim);
    return make_strided_view(
        self,
        sizes,
        self.strides(),
        offset);
}

std::vector<at::Tensor> split_with_sizes_impl(
    const at::Tensor &self,
    const std::vector<int64_t> &split_sizes,
    int64_t dim)
{
    check_split_input(self);
    const int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
    validate_split_sizes(self.size(wrapped_dim), split_sizes);

    std::vector<at::Tensor> outputs;
    outputs.reserve(split_sizes.size());
    int64_t start = 0;
    for (const int64_t length : split_sizes)
    {
        outputs.push_back(
            narrow_view(self, wrapped_dim, start, length));
        start += length;
    }
    return outputs;
}

} // namespace

std::vector<at::Tensor> split_with_sizes(
    const at::Tensor &self,
    c10::SymIntArrayRef split_sizes,
    int64_t dim)
{
    return split_with_sizes_impl(
        self,
        symint_array_to_int64(split_sizes),
        dim);
}

std::vector<at::Tensor> split_sizes_array(
    const at::Tensor &self,
    at::IntArrayRef split_sizes,
    int64_t dim)
{
    std::vector<int64_t> sizes(split_sizes.begin(), split_sizes.end());
    return split_with_sizes_impl(self, sizes, dim);
}

std::vector<at::Tensor> split_tensor(
    const at::Tensor &self,
    c10::SymInt split_size,
    int64_t dim)
{
    check_split_input(self);
    const int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
    const std::vector<int64_t> sizes = compute_equal_split_sizes(
        self.size(wrapped_dim),
        split_size.expect_int());
    return split_with_sizes_impl(self, sizes, wrapped_dim);
}

std::vector<at::Tensor> chunk(
    const at::Tensor &self,
    int64_t chunks,
    int64_t dim)
{
    check_split_input(self);
    const int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
    const std::vector<int64_t> sizes =
        compute_chunk_sizes(self.size(wrapped_dim), chunks);
    return split_with_sizes_impl(self, sizes, wrapped_dim);
}

} // namespace torch_nntile
