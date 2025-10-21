/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/scale_slice.cc
 * Tensor wrappers for scaling of a broadcasted slice
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/scale_slice.hh"
#include "nntile/tile/scale_slice.hh"

namespace nntile::tensor
{

//! Asynchronous tensor scaling of a broadcasted slice
template<typename T>
void scale_slice_async(Scalar alpha, const Tensor<T> &src, const Tensor<T> &dst, Index axis)
{
    // Check inputs
    if(src.ndim != 2)
    {
        throw std::runtime_error("src.ndim != 2");
    }
    if(dst.ndim != 3)
    {
        throw std::runtime_error("dst.ndim != 3");
    }
    if(axis < 0 or axis >= dst.ndim)
    {
        throw std::runtime_error("axis < 0 or axis >= dst.ndim");
    }
    // Check dimensions
    if(src.shape[0] != dst.shape[0])
    {
        throw std::runtime_error("src.shape[0] != dst.shape[0]");
    }
    if(src.shape[1] != dst.shape[2])
    {
        throw std::runtime_error("src.shape[1] != dst.shape[2]");
    }
    // Check axis
    if(axis != 1)
    {
        throw std::runtime_error("axis != 1");
    }
    // Check strides
    if(src.stride[0] != 1)
    {
        throw std::runtime_error("src.stride[0] != 1");
    }
    if(dst.stride[0] != 1)
    {
        throw std::runtime_error("dst.stride[0] != 1");
    }
    // Get dimensions
    Index m = src.shape[0];
    Index n = src.shape[1];
    Index k = dst.shape[1];
    // Submit task
    tile::scale_slice_async<T>(alpha, src, dst, axis);
}

//! Blocking version of tensor scaling of a broadcasted slice
template<typename T>
void scale_slice(Scalar alpha, const Tensor<T> &src, const Tensor<T> &dst, Index axis)
{
    scale_slice_async<T>(alpha, src, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void scale_slice_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst, Index axis);

template
void scale_slice_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst, Index axis);

template
void scale_slice_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src, const Tensor<fp32_fast_tf32_t> &dst, Index axis);

template
void scale_slice_async<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src, const Tensor<fp32_fast_fp16_t> &dst, Index axis);

template
void scale_slice_async<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src, const Tensor<fp32_fast_bf16_t> &dst, Index axis);

template
void scale_slice_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src, const Tensor<bf16_t> &dst, Index axis);

template
void scale_slice_async<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src, const Tensor<fp16_t> &dst, Index axis);

template
void scale_slice<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst, Index axis);

template
void scale_slice<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst, Index axis);

template
void scale_slice<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src, const Tensor<fp32_fast_tf32_t> &dst, Index axis);

template
void scale_slice<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src, const Tensor<fp32_fast_fp16_t> &dst, Index axis);

template
void scale_slice<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src, const Tensor<fp32_fast_bf16_t> &dst, Index axis);

template
void scale_slice<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src, const Tensor<bf16_t> &dst, Index axis);

template
void scale_slice<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src, const Tensor<fp16_t> &dst, Index axis);

} // namespace nntile::tensor
