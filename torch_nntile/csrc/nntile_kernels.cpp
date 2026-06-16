/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_kernels.cpp
 */

#include "nntile_allocator.h"

#include <ATen/EmptyTensor.h>
#include <ATen/InferSize.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/Resize.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/ScalarType.h>
#include <torch/library.h>

#include "nntile_cpu_fallback.h"

#include <cstring>
#include <optional>

namespace torch_nntile
{

namespace
{

constexpr c10::DispatchKeySet kPrivateUse1DispatchKeySet(
    c10::DispatchKey::PrivateUse1);

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

bool is_nntile_or_cpu(const at::Tensor &tensor)
{
    return tensor.is_cpu() || is_nntile_device(tensor.device());
}

void check_copy_devices(const at::Tensor &self, const at::Tensor &dst)
{
    TORCH_CHECK(
        is_nntile_or_cpu(self) && is_nntile_or_cpu(dst),
        "nntile stub copy supports CPU <-> nntile only");
    TORCH_CHECK(self.sizes() == dst.sizes(), "copy size mismatch");
    TORCH_CHECK(
        self.scalar_type() == dst.scalar_type(),
        "copy dtype mismatch");
    TORCH_CHECK(
        self.is_contiguous() && dst.is_contiguous(),
        "nntile stub copy requires contiguous tensors");
}

void memcpy_tensors(const at::Tensor &src, at::Tensor &dst)
{
    const std::size_t nbytes = static_cast<std::size_t>(src.nbytes());
    if (nbytes == 0)
    {
        return;
    }
    std::memcpy(
        dst.storage().data_ptr().get(),
        src.storage().data_ptr().get(),
        nbytes);
}

at::Scalar tensor_to_scalar(const at::Tensor &self)
{
    if (self.scalar_type() == at::ScalarType::Float)
    {
        return *self.data_ptr<float>();
    }
    if (self.scalar_type() == at::ScalarType::Double)
    {
        return *self.data_ptr<double>();
    }
    if (self.scalar_type() == at::ScalarType::Int)
    {
        return *self.data_ptr<int32_t>();
    }
    if (self.scalar_type() == at::ScalarType::Long)
    {
        return *self.data_ptr<int64_t>();
    }
    TORCH_CHECK(false, "Unsupported dtype for _local_scalar_dense on nntile");
    return at::Scalar();
}

} // namespace

at::Tensor empty_memory_format(
    at::IntArrayRef size,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<at::MemoryFormat> memory_format_opt)
{
    const auto device = c10::device_or_default(device_opt);
    TORCH_CHECK(is_nntile_device(device), "empty.memory_format: expected nntile");
    TORCH_CHECK(
        c10::layout_or_default(layout_opt) == c10::Layout::Strided,
        "Non-strided layout not supported on nntile stub");
    TORCH_CHECK(
        !c10::pinned_memory_or_default(pin_memory_opt),
        "Pin memory is CPU-only");
    const c10::DeviceGuard device_guard(device);
    return at::detail::empty_generic(
        size,
        get_nntile_allocator(),
        kPrivateUse1DispatchKeySet,
        c10::dtype_or_default(dtype_opt),
        memory_format_opt);
}

at::Tensor empty_strided(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt)
{
    const auto device = c10::device_or_default(device_opt);
    TORCH_CHECK(is_nntile_device(device), "empty_strided: expected nntile");
    TORCH_CHECK(
        c10::layout_or_default(layout_opt) == c10::Layout::Strided,
        "Non-strided layout not supported on nntile stub");
    TORCH_CHECK(
        !c10::pinned_memory_or_default(pin_memory_opt),
        "Pin memory is CPU-only");
    const c10::DeviceGuard device_guard(device);
    return at::detail::empty_strided_generic(
        size,
        stride,
        get_nntile_allocator(),
        kPrivateUse1DispatchKeySet,
        c10::dtype_or_default(dtype_opt));
}

at::Tensor as_strided(
    const at::Tensor &self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    std::optional<int64_t> storage_offset)
{
    TORCH_CHECK(is_nntile_device(self.device()), "as_strided: expected nntile");
    const int64_t storage_offset_value =
        storage_offset.value_or(self.storage_offset());
    at::Tensor result = at::detail::make_tensor<at::TensorImpl>(
        c10::Storage(self.storage()),
        self.key_set(),
        self.dtype());
    auto *result_impl = result.unsafeGetTensorImpl();
    result_impl->set_storage_offset(storage_offset_value);
    result_impl->set_sizes_and_strides(size, stride);
    return result;
}

at::Tensor reshape_alias(
    const at::Tensor &self,
    at::IntArrayRef size,
    at::IntArrayRef stride)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "_reshape_alias: expected nntile");
    at::Tensor result = at::detail::make_tensor<at::TensorImpl>(
        c10::Storage(self.storage()),
        self.key_set(),
        self.dtype());
    auto *result_impl = result.unsafeGetTensorImpl();
    result_impl->set_storage_offset(self.storage_offset());
    result_impl->set_sizes_and_strides(size, stride);
    return result;
}

at::Tensor view(const at::Tensor &self, at::IntArrayRef size)
{
    TORCH_CHECK(is_nntile_device(self.device()), "view: expected nntile");
    const auto inferred = at::infer_size_dv(size, self.numel());
    const auto stride = at::detail::computeStride(
        self.sizes(),
        self.strides(),
        inferred);
    TORCH_CHECK(
        stride.has_value(),
        "view size is not compatible with input tensor's size and stride");
    return reshape_alias(self, inferred, *stride);
}

const at::Tensor &resize_(
    const at::Tensor &self,
    c10::SymIntArrayRef size,
    std::optional<at::MemoryFormat> memory_format)
{
    TORCH_CHECK(is_nntile_device(self.device()), "resize_: expected nntile");
    if (memory_format.has_value())
    {
        TORCH_CHECK(
            *memory_format == at::MemoryFormat::Contiguous,
            "nntile stub resize_ supports contiguous layout only");
    }
    std::vector<int64_t> sizes;
    sizes.reserve(size.size());
    for (const auto &dim : size)
    {
        sizes.push_back(dim.guard_int(__FILE__, __LINE__));
    }
    at::native::resize_impl_cpu_(
        self.unsafeGetTensorImpl(),
        sizes,
        std::nullopt);
    return self;
}

at::Tensor copy_from(
    const at::Tensor &self,
    const at::Tensor &dst,
    bool /*non_blocking*/)
{
    check_copy_devices(self, dst);
    at::Tensor mutable_dst = dst;
    memcpy_tensors(self, mutable_dst);
    return dst;
}

at::Tensor copy_from_and_resize(const at::Tensor &self, const at::Tensor &dst)
{
    if (self.sizes() != dst.sizes())
    {
        resize_(dst, c10::SymIntArrayRef(self.sym_sizes()), std::nullopt);
    }
    return copy_from(self, dst, false);
}

at::Scalar local_scalar_dense(const at::Tensor &self)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "_local_scalar_dense: expected nntile");
    TORCH_CHECK(self.numel() > 0, "Cannot convert empty tensor to scalar");
    return tensor_to_scalar(self);
}

at::Tensor &set_source_tensor(at::Tensor &result, const at::Tensor &source)
{
    TORCH_CHECK(
        is_nntile_device(result.device()),
        "set_.source_Tensor: expected nntile result");
    TORCH_CHECK(
        is_nntile_device(source.device()),
        "set_.source_Tensor: expected nntile source");
    result.unsafeGetTensorImpl()->set_storage_offset(source.storage_offset());
    result.unsafeGetTensorImpl()->set_sizes_and_strides(
        source.sizes(),
        source.strides());
    return result;
}

at::Tensor &set_source_storage(at::Tensor &result, at::Storage src)
{
    TORCH_CHECK(
        is_nntile_device(result.device()),
        "set_.source_Storage: expected nntile");
    const int64_t new_size = static_cast<int64_t>(
        src.nbytes() / result.dtype().itemsize());
    result.unsafeGetTensorImpl()->set_storage_offset(0);
    at::native::resize_impl_cpu_(
        result.unsafeGetTensorImpl(),
        new_size,
        std::nullopt,
        /*resize_storage=*/false);
    result.unsafeGetTensorImpl()->set_storage_keep_dtype(std::move(src));
    return result;
}

at::Tensor &set_source_storage_storage_offset(
    at::Tensor &result,
    at::Storage src,
    int64_t storage_offset,
    at::IntArrayRef size,
    at::IntArrayRef stride)
{
    TORCH_CHECK(
        is_nntile_device(result.device()),
        "set_.source_Storage_storage_offset: expected nntile");
    result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
    result.unsafeGetTensorImpl()->set_storage_keep_dtype(std::move(src));
    result.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
    return result;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("empty.memory_format", TORCH_FN(torch_nntile::empty_memory_format));
    m.impl("empty_strided", TORCH_FN(torch_nntile::empty_strided));
    m.impl("as_strided", TORCH_FN(torch_nntile::as_strided));
    m.impl("view", TORCH_FN(torch_nntile::view));
    m.impl("_reshape_alias", TORCH_FN(torch_nntile::reshape_alias));
    m.impl("resize_", TORCH_FN(torch_nntile::resize_));
    m.impl("_copy_from", TORCH_FN(torch_nntile::copy_from));
    m.impl("_copy_from_and_resize", TORCH_FN(torch_nntile::copy_from_and_resize));
    m.impl("_local_scalar_dense", TORCH_FN(torch_nntile::local_scalar_dense));
    m.impl("set_.source_Tensor", TORCH_FN(torch_nntile::set_source_tensor));
    m.impl("set_.source_Storage", TORCH_FN(torch_nntile::set_source_storage));
    m.impl(
        "set_.source_Storage_storage_offset",
        TORCH_FN(torch_nntile::set_source_storage_storage_offset));
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m)
{
    m.fallback(torch::CppFunction::makeFromBoxedFunction<
               &torch_nntile::cpu_fallback>());
}
