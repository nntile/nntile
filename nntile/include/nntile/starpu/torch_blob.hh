/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/starpu/torch_blob.hh
 * from_blob helpers for torch-native StarPU codelets (CPU / CUDA).
 *
 * @version 1.1.0
 */

#pragma once

#include <nntile/defs.h>

#ifndef NNTILE_TORCH_NATIVE_OPS
#error "nntile/starpu/torch_blob.hh requires NNTILE_TORCH_NATIVE_OPS"
#endif

#include <cstdint>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/ops/from_blob.h>
#include <c10/util/Optional.h>

#include <nntile/base_types.hh>
#include <nntile/core/torch_meta.hh>

namespace nntile::starpu::torch_blob
{

//! Thread-local default device for from_blob (CPU, or CUDA under TorchCudaEnv).
inline at::Device &default_device_tls()
{
    thread_local at::Device device = at::kCPU;
    return device;
}

inline std::vector<std::int64_t> to_i64(const Index *data, Index n)
{
    std::vector<std::int64_t> out(static_cast<size_t>(n));
    for (Index i = 0; i < n; ++i)
    {
        out[static_cast<size_t>(i)] =
            static_cast<std::int64_t>(data[i]);
    }
    return out;
}

inline std::vector<std::int64_t> to_i64(
    const std::vector<Index> &data)
{
    return to_i64(data.data(), static_cast<Index>(data.size()));
}

inline at::Device resolve_device(
    const c10::optional<at::Device> &device)
{
    return device.has_value() ? *device : default_device_tls();
}

//! Empty deleter: StarPU owns the buffer.
inline at::Tensor blob_fp32(
    float *ptr,
    const std::vector<std::int64_t> &sizes,
    const std::vector<std::int64_t> &strides,
    c10::optional<at::Device> device = c10::nullopt)
{
    auto opts = at::TensorOptions()
        .dtype(at::kFloat)
        .device(resolve_device(device));
    return at::from_blob(
        ptr,
        at::IntArrayRef(sizes),
        at::IntArrayRef(strides),
        /*deleter=*/[](void *) {},
        opts);
}

inline at::Tensor blob_fp32(
    float *ptr,
    const core::TorchTileMeta &meta,
    c10::optional<at::Device> device = c10::nullopt)
{
    return blob_fp32(
        ptr,
        to_i64(meta.sizes),
        to_i64(meta.strides),
        device);
}

inline at::Tensor blob_i64(
    std::int64_t *ptr,
    const std::vector<std::int64_t> &sizes,
    const std::vector<std::int64_t> &strides,
    c10::optional<at::Device> device = c10::nullopt)
{
    auto opts = at::TensorOptions()
        .dtype(at::kLong)
        .device(resolve_device(device));
    return at::from_blob(
        ptr,
        at::IntArrayRef(sizes),
        at::IntArrayRef(strides),
        /*deleter=*/[](void *) {},
        opts);
}

inline at::Tensor blob_bool(
    bool *ptr,
    const std::vector<std::int64_t> &sizes,
    const std::vector<std::int64_t> &strides,
    c10::optional<at::Device> device = c10::nullopt)
{
    auto opts = at::TensorOptions()
        .dtype(at::kBool)
        .device(resolve_device(device));
    return at::from_blob(
        ptr,
        at::IntArrayRef(sizes),
        at::IntArrayRef(strides),
        /*deleter=*/[](void *) {},
        opts);
}

} // namespace nntile::starpu::torch_blob
