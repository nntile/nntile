/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/tests/parity_helpers.hh
 * Shared helpers for libtorch_nntile C++ CPU vs PrivateUse1 parity tests.
 */

#pragma once

#include <torch/torch.h>

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace torch_nntile::test
{

inline c10::Device nntile_device()
{
    return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

inline at::Tensor to_cpu(at::Tensor const &t)
{
    return t.detach().cpu().contiguous();
}

inline void assert_close(
    at::Tensor const &got,
    at::Tensor const &ref,
    double rtol = 1e-4,
    double atol = 1e-4,
    char const *what = "tensor")
{
    at::Tensor a = to_cpu(got).to(at::kFloat);
    at::Tensor b = to_cpu(ref).to(at::kFloat);
    if (a.sizes() != b.sizes())
    {
        throw std::runtime_error(
            std::string(what) + ": shape mismatch");
    }
    at::Tensor diff = (a - b).abs();
    at::Tensor tol = atol + rtol * b.abs();
    if ((diff > tol).any().item<bool>())
    {
        double max_diff = diff.max().item<double>();
        throw std::runtime_error(
            std::string(what) + ": max abs diff " +
            std::to_string(max_diff));
    }
}

//! Forward + backward parity for a unary/binary ATen op on PrivateUse1.
template <typename Op>
void assert_op_forward_backward(
    Op op,
    std::vector<at::Tensor> inputs_cpu,
    double rtol = 1e-4,
    double atol = 1e-4,
    double bwd_rtol = 1e-3,
    double bwd_atol = 1e-3)
{
    std::vector<at::Tensor> cpu_inputs;
    cpu_inputs.reserve(inputs_cpu.size());
    for (at::Tensor const &t : inputs_cpu)
    {
        if (t.is_floating_point())
        {
            cpu_inputs.push_back(
                t.detach().clone().set_requires_grad(t.requires_grad()));
        }
        else
        {
            cpu_inputs.push_back(t.detach().clone());
        }
    }

    at::Tensor y_ref = op(cpu_inputs);
    at::Tensor grad = torch::randn_like(y_ref);
    y_ref.backward(grad);

    std::vector<at::Tensor> nnt_inputs;
    nnt_inputs.reserve(inputs_cpu.size());
    c10::Device const dev = nntile_device();
    for (at::Tensor const &t : inputs_cpu)
    {
        if (t.is_floating_point())
        {
            nnt_inputs.push_back(
                t.detach().contiguous().to(dev).set_requires_grad(
                    t.requires_grad()));
        }
        else
        {
            nnt_inputs.push_back(t.detach().contiguous().to(dev));
        }
    }

    at::Tensor y_nnt = op(nnt_inputs);
    assert_close(y_nnt, y_ref, rtol, atol, "forward");

    // Gradients live on PrivateUse1; LibTorch autograd needs the nntile
    // PrivateUse1 backend + DeviceGuard registered (see nntile_hooks.cpp).
    at::Tensor grad_nnt = grad.contiguous().to(dev);
    y_nnt.backward(grad_nnt, /*keep_graph=*/true, /*create_graph=*/false);

    for (std::size_t i = 0; i < cpu_inputs.size(); ++i)
    {
        if (!cpu_inputs[i].requires_grad())
        {
            continue;
        }
        assert_close(
            nnt_inputs[i].grad(),
            cpu_inputs[i].grad(),
            bwd_rtol,
            bwd_atol,
            "backward");
    }
}

} // namespace torch_nntile::test
