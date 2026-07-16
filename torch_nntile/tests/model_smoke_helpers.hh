/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/tests/model_smoke_helpers.hh
 * Helpers for C++ model forward/backward smoke (shape/device only).
 */

#pragma once

#include "parity_helpers.hh"

#include <torch_nntile/module_to.hh>
#include <torch_nntile/runtime.hh>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace torch_nntile::test
{

struct ContextGuard
{
    ContextGuard()
    {
        if (!torch_nntile::is_context_initialized())
        {
            torch_nntile::init_context(
                1,
                0,
                0,
                "/tmp/nntile_ooc",
                16ull * 1024ull * 1024ull,
                0,
                0,
                false);
            torch_nntile::restrict_cpu();
        }
    }

    ~ContextGuard()
    {
        if (torch_nntile::is_context_initialized())
        {
            torch_nntile::wait_for_all();
            torch_nntile::reset_graph_session();
        }
    }
};

inline at::Tensor bool_causal_mask(int64_t seq)
{
    auto opts = torch::TensorOptions()
        .dtype(torch::kBool)
        .device(torch::kCPU);
    auto q = torch::arange(
        seq,
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
        .unsqueeze(1);
    auto k = torch::arange(
        seq,
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
        .unsqueeze(0);
    return (k <= q).to(opts);
}

inline at::Tensor bool_local_causal_mask(
    int64_t seq,
    int64_t window)
{
    auto opts = torch::TensorOptions()
        .dtype(torch::kBool)
        .device(torch::kCPU);
    auto q = torch::arange(
        seq,
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
        .unsqueeze(1);
    auto k = torch::arange(
        seq,
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
        .unsqueeze(0);
    return ((k <= q) & ((q - k) < window)).to(opts);
}

//! Host RoPE tables ``[batch, seq, head_dim/2]`` (matches Llama warm cache).
inline void rope_sin_cos(
    int64_t batch,
    int64_t seq,
    int64_t head_dim,
    double rope_theta,
    at::Tensor &sin_out,
    at::Tensor &cos_out)
{
    if (head_dim % 2 != 0)
    {
        throw std::invalid_argument("rope: head_dim must be even");
    }
    int64_t const half = head_dim / 2;
    std::vector<float> inv(static_cast<std::size_t>(half));
    for (int64_t i = 0; i < half; ++i)
    {
        double idx = static_cast<double>(2 * i);
        inv[static_cast<std::size_t>(i)] = static_cast<float>(
            1.0 /
            std::pow(
                rope_theta,
                idx / static_cast<double>(head_dim)));
    }
    auto opts = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCPU);
    sin_out = torch::empty({batch, seq, half}, opts);
    cos_out = torch::empty_like(sin_out);
    auto sin_a = sin_out.accessor<float, 3>();
    auto cos_a = cos_out.accessor<float, 3>();
    for (int64_t b = 0; b < batch; ++b)
    {
        for (int64_t s = 0; s < seq; ++s)
        {
            for (int64_t h = 0; h < half; ++h)
            {
                double angle = static_cast<double>(s) *
                    static_cast<double>(
                        inv[static_cast<std::size_t>(h)]);
                sin_a[b][s][h] = static_cast<float>(std::sin(angle));
                cos_a[b][s][h] = static_cast<float>(std::cos(angle));
            }
        }
    }
}

//! Shape/device smoke: forward + one autograd step over float inputs/params.
inline void assert_module_fwd_bwd_smoke(
    torch::nn::Module &module,
    std::function<at::Tensor()> forward_fn,
    std::vector<at::Tensor> float_inputs,
    c10::IntArrayRef expected_shape,
    bool require_backward = true)
{
    c10::Device const dev = nntile_device();
    module_to_device(module, dev);

    at::Tensor y = forward_fn();
    REQUIRE(y.defined());
    REQUIRE(y.device().type() == c10::DeviceType::PrivateUse1);
    REQUIRE(y.sizes() == expected_shape);

    if (!require_backward)
    {
        return;
    }

    std::vector<at::Tensor> targets;
    targets.reserve(float_inputs.size() + 64);
    for (at::Tensor &t : float_inputs)
    {
        if (t.defined() && t.requires_grad())
        {
            targets.push_back(t);
        }
    }
    for (auto &p : module.parameters(/*recurse=*/true))
    {
        if (p.requires_grad())
        {
            targets.push_back(p);
        }
    }
    REQUIRE_FALSE(targets.empty());

    at::Tensor gout = torch::ones_like(y);
    // allow_unused: some registered params may be cache/aux tensors.
    auto grads = torch::autograd::grad(
        /*outputs=*/{y},
        /*inputs=*/targets,
        /*grad_outputs=*/{gout},
        /*retain_graph=*/false,
        /*create_graph=*/false,
        /*allow_unused=*/true);
    REQUIRE(grads.size() == targets.size());
    int used = 0;
    for (std::size_t i = 0; i < grads.size(); ++i)
    {
        if (!grads[i].defined())
        {
            continue;
        }
        ++used;
        REQUIRE(
            grads[i].device().type() == c10::DeviceType::PrivateUse1);
        REQUIRE(grads[i].sizes() == targets[i].sizes());
    }
    REQUIRE(used > 0);
}

inline at::Tensor to_nntile_float(at::Tensor t, bool requires_grad)
{
    auto out = t.detach().contiguous().to(nntile_device());
    if (requires_grad)
    {
        out = out.set_requires_grad(true);
    }
    return out;
}

inline at::Tensor to_nntile_long(at::Tensor t)
{
    return t.detach().contiguous().to(nntile_device());
}

} // namespace torch_nntile::test
