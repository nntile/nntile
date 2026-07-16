/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_module_to.h
 * Move torch::nn::Module params/buffers onto PrivateUse1 (nntile).
 *
 * LibTorch ``Module::to`` uses ``Variable::set_data``, which rejects
 * cross-device-type updates. Python ``nn.Module.to`` replaces Parameters
 * instead; this helper rematerializes TensorImpl the same way.
 */

#pragma once

#include <torch/torch.h>

namespace torch_nntile
{

inline void tensor_rematerialize_on_device(
    torch::Tensor& t,
    torch::Device const& device)
{
    if (!t.defined() || t.device() == device)
    {
        return;
    }
    bool requires_grad = t.requires_grad();
    auto moved = t.detach().to(device);
    t.unsafeGetTensorImpl()->shallow_copy_from(moved.getIntrusivePtr());
    if (requires_grad != t.requires_grad())
    {
        t.set_requires_grad(requires_grad);
    }
}

inline void module_to_device(
    torch::nn::Module& module,
    torch::Device const& device)
{
    torch::NoGradGuard guard;
    for (auto& param : module.parameters(/*recurse=*/false))
    {
        tensor_rematerialize_on_device(param, device);
    }
    for (auto& buffer : module.buffers(/*recurse=*/false))
    {
        tensor_rematerialize_on_device(buffer, device);
    }
    for (auto& child : module.children())
    {
        module_to_device(*child, device);
    }
}

} // namespace torch_nntile
