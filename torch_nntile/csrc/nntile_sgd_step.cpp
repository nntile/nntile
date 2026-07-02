/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_sgd_step.cpp
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"
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

void check_sgd_step_tensors(
    const at::Tensor &param,
    const at::Tensor &grad,
    const at::Tensor &velocity)
{
    TORCH_CHECK(
        is_nntile_device(param.device()),
        "nntile sgd_step: param must be on device nntile");
    TORCH_CHECK(
        is_nntile_device(grad.device()),
        "nntile sgd_step: grad must be on device nntile");
    TORCH_CHECK(
        is_nntile_device(velocity.device()),
        "nntile sgd_step: velocity must be on device nntile");
    TORCH_CHECK(
        param.scalar_type() == at::ScalarType::Float,
        "nntile sgd_step supports float32 only");
    TORCH_CHECK(
        grad.scalar_type() == at::ScalarType::Float,
        "nntile sgd_step supports float32 only");
    TORCH_CHECK(
        velocity.scalar_type() == at::ScalarType::Float,
        "nntile sgd_step supports float32 only");
    TORCH_CHECK(
        param.sizes() == grad.sizes() && grad.sizes() == velocity.sizes(),
        "nntile sgd_step: param, grad, velocity shapes must match");
    TORCH_CHECK(
        param.is_contiguous() && grad.is_contiguous() &&
            velocity.is_contiguous(),
        "nntile sgd_step requires contiguous tensors");
}

} // namespace

void sgd_step(
    at::Tensor &param,
    const at::Tensor &grad,
    at::Tensor &velocity,
    int64_t num_iter,
    double lr,
    double momentum,
    double weight_decay,
    double dampening,
    bool nesterov)
{
    check_sgd_step_tensors(param, grad, velocity);
    TORCH_CHECK(num_iter >= 1, "nntile sgd_step: num_iter must be >= 1");
    mark_staged_input_tensor(param);
    mark_staged_input_tensor(grad);
    mark_staged_input_tensor(velocity);
    pin_graph_op_inputs({param, grad, velocity});
    tensor_sgd_step_fp32(
        num_iter,
        static_cast<float>(momentum),
        static_cast<float>(lr),
        static_cast<float>(weight_decay),
        static_cast<float>(dampening),
        nesterov,
        grad,
        velocity,
        param);
}

} // namespace torch_nntile
