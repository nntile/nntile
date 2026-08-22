/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_mse_loss.cpp
 */

#include "nntile_mse_loss.h"

#include "nntile_executor.h"
#include "nntile_tensor_gc.h"

#include <ATen/TensorUtils.h>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_mse_loss_input(const at::Tensor &x)
{
    TORCH_CHECK(
        is_nntile_device(x.device()),
        "nntile mse_loss: expected nntile x");
    TORCH_CHECK(
        x.scalar_type() == at::ScalarType::Float,
        "nntile mse_loss supports float32 only");
    TORCH_CHECK(x.is_contiguous(), "nntile mse_loss requires contiguous");
    TORCH_CHECK(x.numel() > 0, "nntile mse_loss: x must be non-empty");
}

} // namespace

at::Tensor mse_loss_forward(const at::Tensor &x, double scale)
{
    nntile::GraphFillScope record;
    check_mse_loss_input(x);
    at::Tensor loss = empty_metadata_tensor({}, at::kFloat, x.device());
    tensor_mse_loss_fp32(x, static_cast<float>(scale), loss);
    return loss;
}

at::Tensor mse_loss_backward(
    const at::Tensor &x,
    double scale,
    bool needs_grad)
{
    nntile::GraphFillScope record;
    check_mse_loss_input(x);
    at::Tensor grad_x;
    if (needs_grad)
    {
        grad_x = at::empty(
            x.sizes(),
            x.options().memory_format(at::MemoryFormat::Contiguous));
        tensor_mse_loss_backward_fp32(
            x,
            static_cast<float>(scale),
            grad_x);
    }
    return grad_x;
}

} // namespace torch_nntile
