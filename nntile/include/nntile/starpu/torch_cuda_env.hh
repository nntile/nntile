/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/starpu/torch_cuda_env.hh
 * Bind ATen CUDA dispatch to StarPU worker stream + cuBLAS handle.
 *
 * @version 1.1.0
 */

#pragma once

#include <nntile/defs.h>

#ifndef NNTILE_TORCH_NATIVE_OPS
#error "nntile/starpu/torch_cuda_env.hh requires NNTILE_TORCH_NATIVE_OPS"
#endif

#ifndef NNTILE_USE_CUDA
#error "nntile/starpu/torch_cuda_env.hh requires NNTILE_USE_CUDA"
#endif

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <starpu.h>
#include <starpu_cublas_v2.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <nntile/starpu/torch_blob.hh>

namespace nntile::starpu
{

//! RAII: StarPU CUDA stream + cuBLAS for torch-native codelets.
//!
//! from_blob tensors are meta + StarPU pointers only — never take
//! stream / handle from the Tensor. Bind ATen to the worker stream via
//! ``getStreamFromExternal`` so aten::*_out enqueues on StarPU's stream.
//! Also ``cublasSetStream`` on StarPU's local handle (same stream).
class TorchCudaEnv
{
public:
    TorchCudaEnv()
        : stream_(starpu_cuda_get_local_stream()),
          handle_(starpu_cublas_get_local_handle()),
          device_index_(static_cast<c10::DeviceIndex>(
              starpu_worker_get_devid(starpu_worker_get_id()))),
          prev_blob_device_(torch_blob::default_device_tls()),
          device_guard_(device_index_),
          stream_guard_(
              at::cuda::getStreamFromExternal(
                  stream_,
                  device_index_))
    {
        cublasSetStream(handle_, stream_);
        torch_blob::default_device_tls() = device();
    }

    ~TorchCudaEnv()
    {
        torch_blob::default_device_tls() = prev_blob_device_;
    }

    TorchCudaEnv(const TorchCudaEnv &) = delete;
    TorchCudaEnv &operator=(const TorchCudaEnv &) = delete;

    cudaStream_t stream() const noexcept
    {
        return stream_;
    }

    cublasHandle_t handle() const noexcept
    {
        return handle_;
    }

    c10::DeviceIndex device_index() const noexcept
    {
        return device_index_;
    }

    at::Device device() const
    {
        return at::Device(at::kCUDA, device_index_);
    }

private:
    cudaStream_t stream_;
    cublasHandle_t handle_;
    c10::DeviceIndex device_index_;
    at::Device prev_blob_device_;
    at::cuda::CUDAGuard device_guard_;
    at::cuda::CUDAStreamGuard stream_guard_;
};

} // namespace nntile::starpu
