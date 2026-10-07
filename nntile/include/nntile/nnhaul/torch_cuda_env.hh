/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/nnhaul/torch_cuda_env.hh
 * Bind ATen CUDA dispatch to the NNHaul worker stream + cuBLAS handle.
 *
 * @version 1.1.0
 */

#pragma once

#include <nntile/defs.h>

#ifndef NNTILE_TORCH_NATIVE_OPS
#error "nntile/nnhaul/torch_cuda_env.hh requires NNTILE_TORCH_NATIVE_OPS"
#endif

#ifndef NNTILE_USE_CUDA
#error "nntile/nnhaul/torch_cuda_env.hh requires NNTILE_USE_CUDA"
#endif

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <nnhaul/nnhaul.hh>

#include <nntile/starpu/torch_blob.hh>

namespace nntile::haul
{

//! RAII: NNHaul CUDA stream + cuBLAS for torch-native codelets.
//!
//! from_blob tensors are meta + NNHaul pointers only - never take
//! stream / handle from the Tensor. Bind ATen to the worker stream via
//! ``getStreamFromExternal`` so aten::*_out enqueues on the NNHaul
//! worker's compute stream. Also ``cublasSetStream`` on the runtime's
//! cuBLAS handle (same stream).
class HaulTorchCudaEnv
{
public:
    HaulTorchCudaEnv()
        : stream_(::nnhaul::cuda_stream()),
          handle_(::nnhaul::cublas_handle()),
          device_index_(static_cast<c10::DeviceIndex>(
              ::nnhaul::device_index())),
          prev_blob_device_(starpu::torch_blob::default_device_tls()),
          device_guard_(device_index_),
          stream_guard_(
              at::cuda::getStreamFromExternal(
                  stream_,
                  device_index_))
    {
        if (stream_ == nullptr || handle_ == nullptr)
        {
            throw std::runtime_error(
                "HaulTorchCudaEnv: NNHaul CUDA stream or cuBLAS "
                "handle is null (valid only on a CUDA worker while "
                "its codelet runs)");
        }
        cublasSetStream(handle_, stream_);
        starpu::torch_blob::default_device_tls() = device();
    }

    ~HaulTorchCudaEnv()
    {
        starpu::torch_blob::default_device_tls() = prev_blob_device_;
    }

    HaulTorchCudaEnv(const HaulTorchCudaEnv &) = delete;
    HaulTorchCudaEnv &operator=(const HaulTorchCudaEnv &) = delete;

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

} // namespace nntile::haul
