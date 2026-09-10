/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_adam_step.cpp
 */

#include "nntile_adam_step.h"

#include "nntile_executor_classic.h"
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

void check_adam_step_tensors(
    const at::Tensor &param,
    const at::Tensor &grad,
    const at::Tensor &first_moment,
    const at::Tensor &second_moment,
    const char *op_name)
{
    TORCH_CHECK(
        is_nntile_device(param.device()),
        op_name, ": param must be on device nntile");
    TORCH_CHECK(
        is_nntile_device(grad.device()),
        op_name, ": grad must be on device nntile");
    TORCH_CHECK(
        is_nntile_device(first_moment.device()),
        op_name, ": first_moment must be on device nntile");
    TORCH_CHECK(
        is_nntile_device(second_moment.device()),
        op_name, ": second_moment must be on device nntile");
    TORCH_CHECK(
        param.scalar_type() == at::ScalarType::Float,
        op_name, " supports float32 only");
    TORCH_CHECK(
        grad.scalar_type() == at::ScalarType::Float,
        op_name, " supports float32 only");
    TORCH_CHECK(
        first_moment.scalar_type() == at::ScalarType::Float,
        op_name, " supports float32 only");
    TORCH_CHECK(
        second_moment.scalar_type() == at::ScalarType::Float,
        op_name, " supports float32 only");
    TORCH_CHECK(
        param.sizes() == grad.sizes() &&
            grad.sizes() == first_moment.sizes() &&
            first_moment.sizes() == second_moment.sizes(),
        op_name,
        ": param, grad, first_moment, second_moment shapes must match");
    TORCH_CHECK(
        param.is_contiguous() && grad.is_contiguous() &&
            first_moment.is_contiguous() &&
            second_moment.is_contiguous(),
        op_name, " requires contiguous tensors");
}

} // namespace

void adam_step(
    at::Tensor &param,
    const at::Tensor &grad,
    at::Tensor &first_moment,
    at::Tensor &second_moment,
    int64_t num_iter,
    double lr,
    double beta_1,
    double beta_2,
    double eps,
    double weight_decay)
{
    nntile::GraphFillScope record;
    check_adam_step_tensors(
        param, grad, first_moment, second_moment, "nntile adam_step");
    TORCH_CHECK(num_iter >= 1, "nntile adam_step: num_iter must be >= 1");
    classic_tensor_adam_step_fp32(
        num_iter,
        static_cast<float>(beta_1),
        static_cast<float>(beta_2),
        static_cast<float>(eps),
        static_cast<float>(lr),
        static_cast<float>(weight_decay),
        grad,
        first_moment,
        second_moment,
        param);
}

void adamw_step(
    at::Tensor &param,
    const at::Tensor &grad,
    at::Tensor &first_moment,
    at::Tensor &second_moment,
    int64_t num_iter,
    double lr,
    double beta_1,
    double beta_2,
    double eps,
    double weight_decay)
{
    nntile::GraphFillScope record;
    check_adam_step_tensors(
        param, grad, first_moment, second_moment, "nntile adamw_step");
    TORCH_CHECK(num_iter >= 1, "nntile adamw_step: num_iter must be >= 1");
    classic_tensor_adamw_step_fp32(
        num_iter,
        static_cast<float>(beta_1),
        static_cast<float>(beta_2),
        static_cast<float>(eps),
        static_cast<float>(lr),
        static_cast<float>(weight_decay),
        grad,
        first_moment,
        second_moment,
        param);
}

} // namespace torch_nntile
