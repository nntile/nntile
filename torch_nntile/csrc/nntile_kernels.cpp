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
#include <ATen/ops/arange.h>
#include <ATen/ops/full.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/csrc/autograd/custom_function.h>
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
    if (is_metadata_only_tensor(self))
    {
        if (self.scalar_type() == at::ScalarType::Float)
        {
            tensor_fill_fp32(self, value.to<float>());
            return;
        }
        // T5 ``torch.ones(..., dtype=long, device=nntile)`` for masks:
        // host-fill then ingress (StarPU fill is fp32-only).
        TORCH_CHECK(
            !static_cast<bool>(tensor_ref(self)),
            "fill_: non-float metadata fill requires an unbound tensor");
        at::Tensor cpu = at::full(
            self.sizes(),
            value,
            self.options().device(at::kCPU));
        init_nntile_input_from_cpu(cpu, self);
        return;
    }
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
    if (is_nntile_device(options.device()))
    {
        const c10::ScalarType dtype = dtype_opt.has_value()
            ? *dtype_opt
            : self.scalar_type();
        at::Tensor result = empty_metadata_tensor(
            self.sizes(),
            dtype,
            options.device());
        // Always FILL (including autograd ones_like(loss) unit scalars).
        // A constant-without-FILL shortcut leaves the StarPU handle
        // uninitialized; torch-native readers (e.g. nll_loss_backward)
        // take STARPU_R and assert. tensor::fill still sets constant_value
        // so classic CE can fold the scale.
        fill_scalar(result, 1);
        return result;
    }
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
    // Match device=cuda: honor requested strides (metadata-only storage).
    // Ignoring strides made structured kernels / empty_strided factories
    // look contiguous and later resize_output when shapes were reinterpreted.
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
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
    // Untiled torch-native path: non-contiguous views are allowed.
    // StarPU codelets receive sizes/strides/offset via TorchDispatchArgs.
    record_view_alias(self, result);
    return result;
}

//! HF RoPE uses ``aten::alias`` for full-dim slices (``q[..., :D]``).
//! Stock alias builds a new TensorImpl without our BackendMeta, so the
//! view loses the packed QKV TensorRef and densify invents a wrong node.
at::Tensor alias(const at::Tensor &self)
{
    TORCH_CHECK(is_nntile_device(self.device()), "alias: expected nntile");
    at::Tensor result = at::detail::make_tensor<at::TensorImpl>(
        c10::Storage(self.storage()),
        self.key_set(),
        self.dtype());
    auto *result_impl = result.unsafeGetTensorImpl();
    result_impl->set_storage_offset(self.storage_offset());
    result_impl->set_sizes_and_strides(self.sizes(), self.strides());
    record_view_alias(self, result);
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
    // Same as view/as_strided: share the parent TensorRef so later
    // ops pack layout against the storage tile (not a fresh empty).
    record_view_alias(self, result);
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
    // reshape_alias records the TensorRef share.
    return reshape_alias(self, inferred, *stride);
}

//! ``reshape`` densifies then calls ``_unsafe_view``; without this
//! kernel the result keeps storage but drops BackendMeta/TensorRef,
//! so later ``cat`` (SplitBackward) reads an uninitialized handle.
at::Tensor unsafe_view(
    const at::Tensor &self,
    c10::SymIntArrayRef size)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "_unsafe_view: expected nntile");
    std::vector<int64_t> sizes;
    sizes.reserve(size.size());
    for (const auto &dim : size)
    {
        sizes.push_back(dim.guard_int(__FILE__, __LINE__));
    }
    return view(self, sizes);
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

at::Tensor as_strided(
    const at::Tensor &self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    std::optional<int64_t> storage_offset);

at::Tensor contiguous(
    const at::Tensor &self,
    at::MemoryFormat memory_format);

//! Host I/O gathers the full ``TensorNode``. Densify views that are
//! not a dense cover of that node from offset 0: contiguous narrow /
//! Split Backward slices share parent storage and would numel-mismatch
//! on egress. Do not call ``contiguous()``: it returns ``self`` for
//! already-contiguous views (including offset/partial covers).
bool needs_densify_for_host_io(const at::Tensor &self)
{
    if (!is_nntile_device(self.device()) ||
        self.scalar_type() != at::ScalarType::Float)
    {
        return false;
    }
    if (!self.is_contiguous() || self.storage_offset() != 0)
    {
        return true;
    }
    nntile::TensorRef binding = tensor_ref(self);
    if (!binding)
    {
        return false;
    }
    return static_cast<int64_t>(binding.get()->nelems()) !=
        self.numel();
}

at::Tensor densify_for_host_io(const at::Tensor &self)
{
    at::Tensor result = at::empty(
        self.sizes(),
        self.options()
            .memory_format(at::MemoryFormat::Contiguous)
            .requires_grad(false));
    tensor_copy_fp32(self, result);
    return result;
}

//! True when ``dst`` is a view of a larger (or strided) logical: copy must
//! write into the parent buffer, not rebind ``TensorRef`` (SSA).
bool needs_copy_into_view(const at::Tensor &dst)
{
    nntile::TensorRef binding = tensor_ref(dst);
    if (!binding)
    {
        return false;
    }
    if (dst.storage_offset() != 0)
    {
        return true;
    }
    if (static_cast<int64_t>(binding.get()->nelems()) != dst.numel())
    {
        return true;
    }
    return !dst.is_contiguous();
}

void copy_into_nntile_view(
    const at::Tensor &src,
    at::Tensor &dst)
{
    TORCH_CHECK(
        is_nntile_device(dst.device()),
        "copy_into_nntile_view: expected nntile dst");
    at::Tensor src_cpu = src.is_cpu()
        ? src.contiguous()
        : gather_nntile_view_to_cpu(src);
    at::Tensor full_cpu = gather_full_logical_to_cpu(dst);
    full_cpu.as_strided(
               dst.sizes(),
               dst.strides(),
               dst.storage_offset())
        .copy_(src_cpu);
    overwrite_bound_nntile_logical_from_cpu(full_cpu, dst);
}

at::Tensor copy_from(
    const at::Tensor &self,
    const at::Tensor &dst,
    bool /*non_blocking*/)
{
    // Untiled views: densify nntile src before host I/O.
    at::Tensor src = self;
    if (needs_densify_for_host_io(self))
    {
        src = densify_for_host_io(self);
    }
    TORCH_CHECK(
        (src.is_cpu() && is_nntile_device(dst.device())) ||
            (is_nntile_device(src.device()) && dst.is_cpu()) ||
            (is_nntile_device(src.device()) &&
                is_nntile_device(dst.device())),
        "nntile stub copy supports CPU <-> nntile only");
    TORCH_CHECK(src.sizes() == dst.sizes(), "copy size mismatch");
    TORCH_CHECK(
        src.scalar_type() == dst.scalar_type(),
        "copy dtype mismatch");

    at::Tensor mutable_dst = dst;
    if (dst.is_cpu() && is_nntile_device(src.device()))
    {
        if (has_graph_session())
        {
            // .cpu() may allocate a strided host buffer matching the
            // nntile view; gather into a contiguous temp then copy_.
            if (!mutable_dst.is_contiguous())
            {
                at::Tensor tmp = at::empty(
                    src.sizes(),
                    mutable_dst.options().memory_format(
                        at::MemoryFormat::Contiguous));
                copy_nntile_tensor_to_cpu(src, tmp);
                mutable_dst.copy_(tmp);
            }
            else
            {
                copy_nntile_tensor_to_cpu(src, mutable_dst);
            }
            return dst;
        }
    }
    if (is_nntile_device(mutable_dst.device()) && src.is_cpu())
    {
        if (needs_copy_into_view(mutable_dst))
        {
            copy_into_nntile_view(src, mutable_dst);
            return dst;
        }
        TORCH_CHECK(
            src.is_contiguous() && mutable_dst.is_contiguous(),
            "nntile stub copy requires contiguous tensors");
        init_nntile_input_from_cpu(src, mutable_dst);
        return dst;
    }
    else if (
        is_nntile_device(src.device()) &&
        is_nntile_device(mutable_dst.device()))
    {
        if (needs_copy_into_view(mutable_dst))
        {
            copy_into_nntile_view(src, mutable_dst);
            return dst;
        }
        nntile::TensorRef src_binding = tensor_ref(src);
        if (src_binding != nullptr &&
            src.sizes() == mutable_dst.sizes() &&
            src.scalar_type() == mutable_dst.scalar_type())
        {
            // Dense full-cover SSA rebind (unchanged).
            attach_tensor_ref(mutable_dst, src_binding);
            return dst;
        }
        TORCH_CHECK(
            false,
            "nntile-to-nntile copy between distinct metadata-only tensors "
            "is unsupported");
    }
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

namespace
{

at::TensorOptions cpu_opts_like(const at::Tensor &out)
{
    return at::TensorOptions()
        .dtype(out.scalar_type())
        .device(at::kCPU)
        .layout(at::kStrided);
}

at::Tensor &arange_fill_out(
    at::Tensor &out,
    const at::Scalar &start,
    const at::Scalar &end,
    const at::Scalar &step)
{
    TORCH_CHECK(
        is_nntile_device(out.device()),
        "arange: expected nntile out");
    at::Tensor cpu = at::arange(start, end, step, cpu_opts_like(out));
    if (cpu.sizes() != out.sizes())
    {
        resize_(
            out,
            c10::SymIntArrayRef(cpu.sym_sizes()),
            std::nullopt);
    }
    copy_from(cpu, out, /*non_blocking=*/false);
    return out;
}

at::Tensor arange_on_nntile(
    const at::Scalar &start,
    const at::Scalar &end,
    const at::Scalar &step,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt)
{
    const auto device = c10::device_or_default(device_opt);
    TORCH_CHECK(is_nntile_device(device), "arange: expected nntile device");
    TORCH_CHECK(
        c10::layout_or_default(layout_opt) == c10::Layout::Strided,
        "arange: non-strided layout not supported on nntile");
    TORCH_CHECK(
        !c10::pinned_memory_or_default(pin_memory_opt),
        "arange: pin memory is CPU-only");
    const c10::ScalarType dtype = dtype_opt.has_value()
        ? *dtype_opt
        : (start.isFloatingPoint() || end.isFloatingPoint() ||
                step.isFloatingPoint()
                ? at::ScalarType::Float
                : at::ScalarType::Long);
    at::Tensor cpu = at::arange(
        start,
        end,
        step,
        at::TensorOptions().dtype(dtype).device(at::kCPU));
    at::Tensor out = empty_metadata_tensor(cpu.sizes(), dtype, device);
    copy_from(cpu, out, /*non_blocking=*/false);
    return out;
}

} // namespace

at::Tensor arange_end(
    const at::Scalar &end,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory)
{
    return arange_on_nntile(
        /*start=*/0,
        end,
        /*step=*/1,
        dtype,
        layout,
        device,
        pin_memory);
}

at::Tensor arange_start(
    const at::Scalar &start,
    const at::Scalar &end,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory)
{
    return arange_on_nntile(
        start,
        end,
        /*step=*/1,
        dtype,
        layout,
        device,
        pin_memory);
}

at::Tensor arange_start_step(
    const at::Scalar &start,
    const at::Scalar &end,
    const at::Scalar &step,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory)
{
    return arange_on_nntile(
        start,
        end,
        step,
        dtype,
        layout,
        device,
        pin_memory);
}

at::Tensor &arange_out(
    const at::Scalar &end,
    at::Tensor &out)
{
    return arange_fill_out(out, /*start=*/0, end, /*step=*/1);
}

at::Tensor &arange_start_out(
    const at::Scalar &start,
    const at::Scalar &end,
    const at::Scalar &step,
    at::Tensor &out)
{
    return arange_fill_out(out, start, end, step);
}

at::Scalar local_scalar_dense(const at::Tensor &self)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "_local_scalar_dense: expected nntile");
    TORCH_CHECK(self.numel() > 0, "Cannot convert empty tensor to scalar");
    // HF ``cache_position[-1]`` is a 1-element select into a larger
    // logical. Gather via the shared view helper (handles partial covers).
    nntile::TensorRef binding = tensor_ref(self);
    if (binding &&
        (self.storage_offset() != 0 ||
         static_cast<int64_t>(binding.get()->nelems()) != self.numel()))
    {
        return tensor_to_scalar(gather_nntile_view_to_cpu(self).reshape({}));
    }
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
    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    std::swap(sizes[static_cast<size_t>(dim0)], sizes[static_cast<size_t>(dim1)]);
    std::swap(
        strides[static_cast<size_t>(dim0)],
        strides[static_cast<size_t>(dim1)]);
    // Zero-copy view (untiled); layout packed at op-record time.
    return torch_nntile::as_strided(
        self,
        sizes,
        strides,
        std::optional<int64_t>(self.storage_offset()));
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
    return torch_nntile::as_strided(
        self,
        sizes,
        strides,
        std::optional<int64_t>(self.storage_offset()));
}

bool is_partial_storage_cover(const at::Tensor &self)
{
    nntile::TensorRef binding = tensor_ref(self);
    if (!binding)
    {
        return false;
    }
    if (self.storage_offset() != 0 ||
        static_cast<int64_t>(binding.get()->nelems()) != self.numel())
    {
        return true;
    }
    // Same-numel reshape view: TensorNode shape still the parent
    // (e.g. View Backward of [B,S,H,D] → [B,S,H*D]).
    const auto &node_shape = binding.get()->shape();
    if (node_shape.size() != static_cast<std::size_t>(self.dim()))
    {
        return true;
    }
    for (int64_t i = 0; i < self.dim(); ++i)
    {
        if (static_cast<int64_t>(
                node_shape[static_cast<std::size_t>(i)]) !=
            self.size(i))
        {
            return true;
        }
    }
    return false;
}

at::Tensor contiguous(
    const at::Tensor &self,
    at::MemoryFormat memory_format)
{
    TORCH_CHECK(is_nntile_device(self.device()), "contiguous: expected nntile");
    TORCH_CHECK(
        memory_format == at::MemoryFormat::Contiguous,
        "nntile contiguous supports Contiguous memory format only");
    // A 1-element view with storage_offset!=0 is often ``is_contiguous``
    // yet still aliases a larger logical; densify those for host I/O.
    // Same-numel reshape views (View Backward of attention heads) also
    // keep the parent TensorNode shape — densify so cat/split see the
    // logical sizes instead of the storage tile (else SplitBackward
    // cats 3×[B,S,H,D] into [B,S,3*n_embd] and resize_output-warns).
    nntile::TensorRef binding = tensor_ref(self);
    if (self.is_contiguous(memory_format) &&
        !is_partial_storage_cover(self))
    {
        return self;
    }
    if (self.scalar_type() == at::ScalarType::Float)
    {
        // Densify into a fresh contiguous buffer. Do not call clone() here:
        // stock clone(memory_format) redispatches to contiguous and would
        // recurse. AutogradPrivateUse1 wraps this via ContiguousFn.
        at::Tensor result = at::empty(
            self.sizes(),
            self.options()
                .memory_format(at::MemoryFormat::Contiguous)
                .requires_grad(false));
        tensor_copy_fp32(self, result);
        return result;
    }
    // Bool / int views (HF masks): gather full logical, apply view on
    // host, scatter a contiguous result.
    TORCH_CHECK(
        binding,
        "nntile contiguous: unbound non-float tensor");
    nntile::TensorGraph::TensorNode *logical = binding.get();
    std::vector<int64_t> full_sizes(
        logical->shape().begin(),
        logical->shape().end());
    if (full_sizes.empty())
    {
        full_sizes.push_back(static_cast<int64_t>(logical->nelems()));
    }
    at::Tensor full_cpu = at::empty(
        full_sizes,
        self.options().device(at::kCPU).memory_format(
            at::MemoryFormat::Contiguous));
    copy_nntile_tensor_to_cpu(self, full_cpu);
    at::Tensor viewed = full_cpu.as_strided(
        self.sizes(),
        self.strides(),
        self.storage_offset());
    at::Tensor contig_cpu = viewed.contiguous();
    at::Tensor out = empty_metadata_tensor(
        contig_cpu.sizes(),
        contig_cpu.scalar_type(),
        self.device());
    init_nntile_input_from_cpu(contig_cpu, out);
    return out;
}

//! Same-shape densify: identity backward (like CloneBackward).
class ContiguousFn : public torch::autograd::Function<ContiguousFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext * /*ctx*/,
        at::Tensor self)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        return contiguous(self, at::MemoryFormat::Contiguous);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext * /*ctx*/,
        torch::autograd::variable_list grad_outputs)
    {
        return {grad_outputs[0]};
    }
};

at::Tensor contiguous_autograd(
    const at::Tensor &self,
    at::MemoryFormat memory_format)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "contiguous: expected nntile");
    TORCH_CHECK(
        memory_format == at::MemoryFormat::Contiguous,
        "nntile contiguous supports Contiguous memory format only");
    // Match PrivateUse1 contiguous: 1-element / narrow views can be
    // ``is_contiguous`` yet still need densify for host scalar I/O.
    if (self.is_contiguous(memory_format) &&
        !is_partial_storage_cover(self))
    {
        return self;
    }
    return ContiguousFn::apply(self);
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
    m.impl("alias", TORCH_FN(torch_nntile::alias));
    m.impl("view", TORCH_FN(torch_nntile::view));
    m.impl("_unsafe_view", TORCH_FN(torch_nntile::unsafe_view));
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
    m.impl("arange", TORCH_FN(torch_nntile::arange_end));
    m.impl("arange.start", TORCH_FN(torch_nntile::arange_start));
    m.impl("arange.start_step", TORCH_FN(torch_nntile::arange_start_step));
    m.impl("arange.out", TORCH_FN(torch_nntile::arange_out));
    m.impl("arange.start_out", TORCH_FN(torch_nntile::arange_start_out));
}

// AutogradPrivateUse1: ContiguousFn densifies under
// AutoDispatchBelowADInplaceOrView and uses identity backward (same
// logical shape). Do not use AutoDispatchBelowAutograd alone — that
// drops requires_grad / grad_fn on the densify result.
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m)
{
    m.impl("contiguous", TORCH_FN(torch_nntile::contiguous_autograd));
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m)
{
    m.fallback(torch::CppFunction::makeFromBoxedFunction<
               &torch_nntile::cpu_fallback>());
}
