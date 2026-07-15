/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_kernels.cpp
 */

#include "nntile_allocator.h"
#include "nntile_context.h"
#include "nntile_executor.h"
#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_gc.h"
#include "nntile_tensor_meta.h"

#include <ATen/EmptyTensor.h>
#include <ATen/InferSize.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/Resize.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/library.h>
#include <torch/version.h>

#include <cstring>
#include <numeric>
#include <optional>
#include <sstream>

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
    // data_ptr() includes storage_offset; storage().data_ptr() does not.
    std::memcpy(dst.data_ptr(), src.data_ptr(), nbytes);
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

void fill_tensor(at::Tensor &self, const at::Scalar &value)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "fill_: expected nntile tensor");
    TORCH_CHECK(self.is_contiguous(), "fill_: requires contiguous tensor");
    const int64_t nelems = self.numel();
    if (nelems == 0)
    {
        return;
    }
#ifdef TORCH_NNTILE_USE_LIBNNTILE
    if (is_metadata_only_tensor(self))
    {
        TORCH_CHECK(
            self.scalar_type() == at::ScalarType::Float,
            "fill_: metadata-only nntile tensors support float32 in graph mode");
        tensor_fill_fp32(self, value.to<float>());
        return;
    }
#endif
    switch (self.scalar_type())
    {
    case at::ScalarType::Float:
    {
        const float fill_value = value.to<float>();
        float *data = self.data_ptr<float>();
        for (int64_t i = 0; i < nelems; ++i)
        {
            data[i] = fill_value;
        }
        break;
    }
    case at::ScalarType::Double:
    {
        const double fill_value = value.to<double>();
        double *data = self.data_ptr<double>();
        for (int64_t i = 0; i < nelems; ++i)
        {
            data[i] = fill_value;
        }
        break;
    }
    case at::ScalarType::Int:
    {
        const int32_t fill_value = value.to<int32_t>();
        int32_t *data = self.data_ptr<int32_t>();
        for (int64_t i = 0; i < nelems; ++i)
        {
            data[i] = fill_value;
        }
        break;
    }
    case at::ScalarType::Long:
    {
        const int64_t fill_value = value.to<int64_t>();
        int64_t *data = self.data_ptr<int64_t>();
        for (int64_t i = 0; i < nelems; ++i)
        {
            data[i] = fill_value;
        }
        break;
    }
    default:
        TORCH_CHECK(false, "fill_: unsupported dtype on nntile");
    }
}

} // namespace

at::Tensor &fill_scalar(at::Tensor &self, const at::Scalar &value)
{
    fill_tensor(self, value);
    return self;
}

at::Tensor empty_metadata_tensor(
    c10::IntArrayRef size,
    c10::ScalarType dtype,
    c10::Device device)
{
    const c10::DeviceGuard device_guard(device);
    c10::Allocator *allocator = get_nntile_allocator();
    c10::DataPtr data_ptr = allocator->allocate(0);
    auto storage = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        0,
        std::move(data_ptr),
        allocator,
        /*resizable=*/true);
    at::Tensor tensor = at::detail::make_tensor<at::TensorImpl>(
        c10::Storage(std::move(storage)),
        kPrivateUse1DispatchKeySet,
        c10::scalarTypeToTypeMeta(dtype));
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    register_metadata_tensor_storage(tensor);
    return tensor;
}

at::Tensor &zero_tensor(at::Tensor &self)
{
    return fill_scalar(self, 0);
}

at::Tensor ones_like(
    const at::Tensor &self,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<at::MemoryFormat> memory_format_opt)
{
    at::TensorOptions options = self.options();
    if (dtype_opt.has_value())
    {
        options = options.dtype(*dtype_opt);
    }
    if (layout_opt.has_value())
    {
        options = options.layout(*layout_opt);
    }
    if (device_opt.has_value())
    {
        options = options.device(*device_opt);
    }
    if (pin_memory_opt.has_value())
    {
        options = options.pinned_memory(*pin_memory_opt);
    }
#ifdef TORCH_NNTILE_USE_LIBNNTILE
    if (is_nntile_device(options.device()))
    {
        const c10::ScalarType dtype = dtype_opt.has_value()
            ? *dtype_opt
            : self.scalar_type();
        at::Tensor result = empty_metadata_tensor(
            self.sizes(),
            dtype,
            options.device());
        // Autograd ones_like(loss) is a unit scalar. Mark it constant without
        // a FILL so CE can fold the scale; non-scalar ones_like still fills.
        if (dtype == at::ScalarType::Float && result.numel() == 1)
        {
            std::vector<nntile::Index> shape;
            shape.reserve(static_cast<std::size_t>(result.dim()));
            for (const auto dim : result.sizes())
            {
                shape.push_back(static_cast<nntile::Index>(dim));
            }
            nntile::TensorGraph::TensorNode *node = get_or_create_data_node(
                result,
                shape,
                nntile::DataType::FP32,
                false);
            node->set_constant_value(static_cast<nntile::Scalar>(1.0));
            node->note_produced();
            return result;
        }
        fill_scalar(result, 1);
        return result;
    }
#endif
    at::MemoryFormat format = at::MemoryFormat::Contiguous;
    if (memory_format_opt.has_value())
    {
        const at::MemoryFormat requested = *memory_format_opt;
        format = requested == at::MemoryFormat::Preserve
            ? self.suggest_memory_format()
            : requested;
    }
    at::Tensor result = at::empty(
        self.sizes(),
        options.memory_format(format));
#ifndef TORCH_NNTILE_USE_LIBNNTILE
    if (is_nntile_device(result.device()) && is_metadata_only_tensor(result))
    {
        ensure_host_staging(result);
    }
#endif
    result.fill_(1);
    return result;
}

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
    const c10::ScalarType dtype = c10::dtype_or_default(dtype_opt);
    at::Tensor tensor = empty_metadata_tensor(size, dtype, device);
#ifndef TORCH_NNTILE_USE_LIBNNTILE
    ensure_host_staging(tensor);
#endif
    return tensor;
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
    const c10::ScalarType dtype = c10::dtype_or_default(dtype_opt);
    at::Tensor tensor = empty_metadata_tensor(size, dtype, device);
#ifndef TORCH_NNTILE_USE_LIBNNTILE
    ensure_host_staging(tensor);
#endif
    return tensor;
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
    TORCH_CHECK(
        result.is_contiguous(),
        "as_strided: non-contiguous layout is not supported on nntile");
#ifdef TORCH_NNTILE_USE_LIBNNTILE
    record_view_alias(self, result);
#endif
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
    at::Tensor result = reshape_alias(self, inferred, *stride);
#ifdef TORCH_NNTILE_USE_LIBNNTILE
    record_view_alias(self, result);
#endif
    return result;
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
    if (dst.is_cpu() && is_nntile_device(self.device()))
    {
#ifdef TORCH_NNTILE_USE_LIBNNTILE
        if (has_graph_session())
        {
            // run() is async; do not wait_for_all here. copy_nntile_tensor_to_cpu
            // / gather compile finishes any pending run before readout.
            copy_nntile_tensor_to_cpu(self, mutable_dst);
            return dst;
        }
#endif
    }
    if (is_nntile_device(mutable_dst.device()) && self.is_cpu())
    {
#ifdef TORCH_NNTILE_USE_LIBNNTILE
        init_nntile_input_from_cpu(self, mutable_dst);
        return dst;
#else
        ensure_host_staging(mutable_dst);
#endif
    }
    else if (
        is_nntile_device(self.device()) &&
        is_nntile_device(mutable_dst.device()))
    {
#ifdef TORCH_NNTILE_USE_LIBNNTILE
        nntile::TensorRef src_binding = tensor_ref(self);
        if (src_binding != nullptr &&
            self.sizes() == mutable_dst.sizes() &&
            self.scalar_type() == mutable_dst.scalar_type())
        {
            attach_tensor_ref(mutable_dst, src_binding);
            return dst;
        }
        TORCH_CHECK(
            false,
            "nntile-to-nntile copy between distinct metadata-only tensors "
            "is unsupported");
#else
        if (!has_host_staging(mutable_dst) && self.nbytes() > 0)
        {
            ensure_host_staging(mutable_dst);
        }
        memcpy_tensors(self, mutable_dst);
#endif
    }
#ifndef TORCH_NNTILE_USE_LIBNNTILE
    if (has_host_staging(self) || has_host_staging(mutable_dst))
    {
        memcpy_tensors(self, mutable_dst);
    }
#endif
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
    require_no_pending_graph(
        "read a scalar from an nntile tensor "
        "(call torch_nntile.compile_graph() and torch_nntile.run() first)");
    wait_for_all();
    if (is_metadata_only_tensor(self))
    {
        at::Tensor cpu_scalar = at::empty({}, self.options().device(at::kCPU));
        copy_nntile_tensor_to_cpu(self, cpu_scalar);
        return tensor_to_scalar(cpu_scalar);
    }
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

at::Tensor transpose_int(const at::Tensor &self, int64_t dim0, int64_t dim1)
{
    TORCH_CHECK(is_nntile_device(self.device()), "transpose: expected nntile");
    const auto ndim = self.dim();
    TORCH_CHECK(ndim >= 2, "nntile transpose expects at least 2D tensors");
    if (dim0 < 0)
    {
        dim0 += ndim;
    }
    if (dim1 < 0)
    {
        dim1 += ndim;
    }
    TORCH_CHECK(
        dim0 >= 0 && dim0 < ndim && dim1 >= 0 && dim1 < ndim,
        "transpose: dimension out of range");
    if (dim0 == dim1)
    {
        return self;
    }
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile transpose supports float32 only");
    TORCH_CHECK(
        self.is_contiguous(),
        "nntile transpose requires contiguous input");
    auto sizes = self.sizes().vec();
    std::swap(sizes[static_cast<size_t>(dim0)], sizes[static_cast<size_t>(dim1)]);
    at::Tensor result = at::empty(
        c10::IntArrayRef(sizes),
        self.options().memory_format(at::MemoryFormat::Contiguous));
    const int64_t numel = self.numel();
    if (numel == 0)
    {
        return result;
    }
#ifdef TORCH_NNTILE_USE_LIBNNTILE
    tensor_swap_two_axes_fp32(self, result, dim0, dim1);
    return result;
#endif
    tensor_swap_two_axes_fp32(self, result, dim0, dim1);
    return result;
}

at::Tensor t(const at::Tensor &self)
{
    TORCH_CHECK(is_nntile_device(self.device()), "t: expected nntile");
    TORCH_CHECK(self.dim() == 2, "t: expected a 2D tensor");
    return transpose_int(self, 0, 1);
}

at::Tensor permute(const at::Tensor &self, at::IntArrayRef dims)
{
    TORCH_CHECK(is_nntile_device(self.device()), "permute: expected nntile");
    const auto ndim = self.dim();
    TORCH_CHECK(
        static_cast<int64_t>(dims.size()) == ndim,
        "permute: number of dims does not match tensor dim");
    std::vector<int64_t> sizes(static_cast<size_t>(ndim));
    std::vector<int64_t> strides(static_cast<size_t>(ndim));
    std::vector<bool> seen(static_cast<size_t>(ndim), false);
    for (int64_t i = 0; i < ndim; ++i)
    {
        int64_t src = dims[static_cast<size_t>(i)];
        if (src < 0)
        {
            src += ndim;
        }
        TORCH_CHECK(src >= 0 && src < ndim, "permute: dimension out of range");
        TORCH_CHECK(!seen[static_cast<size_t>(src)], "permute: duplicate dim");
        seen[static_cast<size_t>(src)] = true;
        sizes[static_cast<size_t>(i)] = self.size(src);
        strides[static_cast<size_t>(i)] = self.stride(src);
    }
    at::Tensor result = reshape_alias(
        self,
        c10::IntArrayRef(sizes),
        c10::IntArrayRef(strides));
    TORCH_CHECK(
        result.is_contiguous(),
        "permute: non-contiguous layout is not supported on nntile; "
        "use transpose for axis swaps");
#ifdef TORCH_NNTILE_USE_LIBNNTILE
    record_view_alias(self, result);
#endif
    return result;
}

at::Tensor contiguous(
    const at::Tensor &self,
    at::MemoryFormat memory_format)
{
    TORCH_CHECK(is_nntile_device(self.device()), "contiguous: expected nntile");
    TORCH_CHECK(
        memory_format == at::MemoryFormat::Contiguous,
        "nntile contiguous supports Contiguous memory format only");
    if (self.is_contiguous(memory_format))
    {
        return self;
    }
    TORCH_CHECK(
        false,
        "aten::contiguous is not supported on device=nntile; ensure tensors are "
        "contiguous before .to('nntile') or use graph layout ops "
        "(transpose, model_transpose, repeat, view)");
}

at::Tensor contiguous_autograd(
    const at::Tensor &self,
    at::MemoryFormat memory_format)
{
    at::AutoDispatchBelowAutograd guard;
    return contiguous(self, memory_format);
}

void cpu_fallback(const c10::OperatorHandle &op, torch::jit::Stack *stack)
{
    if (!torch_nntile::is_cpu_fallback_enabled())
    {
        std::ostringstream message;
        message << "Operator '" << op.schema().operator_name()
                << "' is not implemented for device nntile and CPU "
                   "fallback is disabled (set cpu_fallback=True in "
                   "torch_nntile.init_context)";
        TORCH_CHECK(false, message.str());
    }
#if (TORCH_VERSION_MAJOR > 2) \
    || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 12)
    at::native::cpu_fallback(
        op,
        stack,
        /*error_on_views=*/false,
        c10::DispatchKey::CPU);
#else
    at::native::cpu_fallback(op, stack);
#endif
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("empty.memory_format", TORCH_FN(torch_nntile::empty_memory_format));
    m.impl("empty_strided", TORCH_FN(torch_nntile::empty_strided));
    m.impl("as_strided", TORCH_FN(torch_nntile::as_strided));
    m.impl("view", TORCH_FN(torch_nntile::view));
    m.impl("_reshape_alias", TORCH_FN(torch_nntile::reshape_alias));
    m.impl("transpose.int", TORCH_FN(torch_nntile::transpose_int));
    m.impl("t", TORCH_FN(torch_nntile::t));
    m.impl("permute", TORCH_FN(torch_nntile::permute));
    m.impl("contiguous", TORCH_FN(torch_nntile::contiguous));
    m.impl("resize_", TORCH_FN(torch_nntile::resize_));
    m.impl("_copy_from", TORCH_FN(torch_nntile::copy_from));
    m.impl("_copy_from_and_resize", TORCH_FN(torch_nntile::copy_from_and_resize));
    m.impl("_local_scalar_dense", TORCH_FN(torch_nntile::local_scalar_dense));
    m.impl("fill_.Scalar", TORCH_FN(torch_nntile::fill_scalar));
    m.impl("zero_", TORCH_FN(torch_nntile::zero_tensor));
    m.impl("ones_like", TORCH_FN(torch_nntile::ones_like));
    m.impl("set_.source_Tensor", TORCH_FN(torch_nntile::set_source_tensor));
    m.impl("set_.source_Storage", TORCH_FN(torch_nntile::set_source_storage));
    m.impl(
        "set_.source_Storage_storage_offset",
        TORCH_FN(torch_nntile::set_source_storage_storage_offset));
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m)
{
    m.impl("contiguous", TORCH_FN(torch_nntile::contiguous_autograd));
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m)
{
    m.fallback(torch::CppFunction::makeFromBoxedFunction<
               &torch_nntile::cpu_fallback>());
}
