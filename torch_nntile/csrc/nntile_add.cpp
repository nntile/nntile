/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_add.cpp
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_gc.h"
#include "nntile_broadcast.h"

#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_add_out_of_place_inputs(
    const at::Tensor &self,
    const at::Tensor &other,
    const std::optional<at::Tensor> &out = std::nullopt)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) &&
            is_nntile_device(other.device()),
        "nntile add expects both operands on device nntile");
    if (out.has_value())
    {
        TORCH_CHECK(
            is_nntile_device(out->device()),
            "nntile add.out expects output on device nntile");
    }
    TORCH_CHECK(
        at::are_expandable(self.sizes(), other.sizes()),
        "nntile add: shape not broadcastable");
    TORCH_CHECK(
        self.scalar_type() == other.scalar_type(),
        "nntile add: dtype mismatch");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile add supports float32 only in phase 2");
    TORCH_CHECK(
        self.is_contiguous() && other.is_contiguous(),
        "nntile add requires contiguous tensors");
    if (out.has_value())
    {
        TORCH_CHECK(
            out->scalar_type() == at::ScalarType::Float,
            "nntile add.out supports float32 only in phase 2");
        TORCH_CHECK(
            out->is_contiguous(),
            "nntile add.out requires contiguous output");
    }
}

void check_add_inplace_inputs(
    const at::Tensor &self,
    const at::Tensor &other)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) &&
            is_nntile_device(other.device()),
        "nntile add expects both operands on device nntile");
    TORCH_CHECK(
        at::are_expandable(other.sizes(), self.sizes()),
        "nntile add_.Tensor: shape not broadcastable to self");
    TORCH_CHECK(
        self.scalar_type() == other.scalar_type(),
        "nntile add: dtype mismatch");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile add supports float32 only in phase 2");
    TORCH_CHECK(
        self.is_contiguous() && other.is_contiguous(),
        "nntile add requires contiguous tensors");
}

at::Tensor broadcast_to_shape(
    const at::Tensor &tensor,
    c10::IntArrayRef target_size)
{
    if (tensor.sizes().equals(target_size) && tensor.is_contiguous())
    {
        return tensor;
    }
    if (tensor.device().type() == c10::DeviceType::PrivateUse1)
    {
#ifdef TORCH_NNTILE_USE_LIBNNTILE
        if (has_pending_graph())
        {
            if (tensor.sizes().equals(target_size))
            {
                return tensor.contiguous();
            }
            const int64_t target_ndim =
                static_cast<int64_t>(target_size.size());
            const int64_t tensor_ndim =
                static_cast<int64_t>(tensor.sizes().size());
            TORCH_CHECK(
                tensor_ndim <= target_ndim,
                "nntile broadcast: tensor rank exceeds target rank");
            std::vector<int64_t> repeats(
                static_cast<std::size_t>(target_ndim),
                1);
            const int64_t pad = target_ndim - tensor_ndim;
            for (int64_t i = 0; i < tensor_ndim; ++i)
            {
                const int64_t in_dim =
                    tensor.sizes()[static_cast<std::size_t>(i)];
                const int64_t out_dim =
                    target_size[static_cast<std::size_t>(i + pad)];
                TORCH_CHECK(
                    in_dim == 1 || in_dim == out_dim,
                    "nntile broadcast: dimension is not broadcastable");
                TORCH_CHECK(
                    out_dim % in_dim == 0,
                    "nntile broadcast: output size is not divisible "
                    "by input");
                repeats[static_cast<std::size_t>(i + pad)] =
                    out_dim / in_dim;
            }
            at::Tensor out = at::empty(
                target_size,
                tensor.options().memory_format(
                    at::MemoryFormat::Contiguous));
            pin_graph_op_inputs({tensor});
            pin_graph_op_output(out, true);
            tensor_repeat_fp32(tensor, out, repeats);
            return out;
        }
#endif
        const at::Tensor cpu_broadcast =
            tensor.cpu().expand(target_size).contiguous();
        at::Tensor out = at::empty(
            target_size,
            tensor.options().memory_format(at::MemoryFormat::Contiguous));
        ensure_host_staging(out);
        out.copy_(cpu_broadcast);
        mark_staged_input_tensor(out);
        return out;
    }
    at::Tensor expanded = tensor.expand(target_size);
    if (!expanded.is_contiguous())
    {
        expanded = expanded.contiguous();
    }
    return expanded;
}

void run_add_kernel(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    const float self_scale = 1.0f;
    const float other_scale = alpha.to<float>();
    pin_graph_op_inputs({self, other});
    pin_graph_op_output(out, false);
    tensor_add_fp32(self_scale, self, other_scale, other, out);
}

} // namespace

at::Tensor add_tensor(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha)
{
    check_add_out_of_place_inputs(self, other);
    const c10::SymIntArrayRef output_size =
        at::infer_size_symdimvector(self.sym_sizes(), other.sym_sizes());
    const at::Tensor lhs =
        broadcast_to_shape(self, C10_AS_INTARRAYREF_SLOW(output_size));
    const at::Tensor rhs =
        broadcast_to_shape(other, C10_AS_INTARRAYREF_SLOW(output_size));
    at::Tensor out = at::empty(
        C10_AS_INTARRAYREF_SLOW(output_size),
        self.options().memory_format(at::MemoryFormat::Contiguous));
    run_add_kernel(lhs, rhs, alpha, out);
    return out;
}

at::Tensor &add_out(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    check_add_out_of_place_inputs(self, other, out);
    const c10::SymIntArrayRef output_size =
        at::infer_size_symdimvector(self.sym_sizes(), other.sym_sizes());
    TORCH_CHECK(
        out.sizes().equals(C10_AS_INTARRAYREF_SLOW(output_size)),
        "nntile add.out: output shape mismatch");
    const at::Tensor lhs =
        broadcast_to_shape(self, C10_AS_INTARRAYREF_SLOW(output_size));
    const at::Tensor rhs =
        broadcast_to_shape(other, C10_AS_INTARRAYREF_SLOW(output_size));
    run_add_kernel(lhs, rhs, alpha, out);
    return out;
}

at::Tensor &add_inplace_tensor(
    at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha)
{
    check_add_inplace_inputs(self, other);
    const at::Tensor other_broadcast =
        broadcast_to_shape(other, self.sizes());
    const float other_scale = alpha.to<float>();
    const float self_scale = 1.0f;
    pin_graph_op_inputs({self, other_broadcast});
    pin_graph_op_output(self, true);
    tensor_add_inplace_fp32(
        other_scale,
        other_broadcast,
        self_scale,
        self);
    return self;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("add.Tensor", TORCH_FN(torch_nntile::add_tensor));
    m.impl("add.out", TORCH_FN(torch_nntile::add_out));
    m.impl("add_.Tensor", TORCH_FN(torch_nntile::add_inplace_tensor));
}
