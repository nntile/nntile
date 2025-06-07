/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/conv2d_bwd_weight_inplace.cc
 * Backward 2D-Convolution of two tensors in WHCN format to get grad of weight
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/conv2d_bwd_weight_inplace.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/conv2d_bwd_weight_inplace.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
Conv2dBwdWeightInplace<std::tuple<T>>::Conv2dBwdWeightInplace():
    codelet("nntile_conv2d_bwd_weight_inplace", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply conv2d_bwd_weight_inplace on StarPU buffer on CPU
template <typename T>
void Conv2dBwdWeightInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    args_t *args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src1 = interfaces[0]->get_ptr<T>();
    const T *src2 = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Launch kernel
    kernel::conv2d_bwd_weight_inplace::cpu<T>(
        args->src1_m,
        args->src1_n,
        args->src1_channels,
        args->batch,
        args->src2_m,
        args->src2_n,
        args->stride_m,
        args->stride_n,
        args->src2_channels,
        args->offset_m,
        args->offset_n,
        args->alpha,
        src1,
        src2,
        args->dst_m,
        args->dst_n,
        args->dilation_m,
        args->dilation_n,
        args->beta,
        dst
    );
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply conv2d_bwd_weight_inplace on StarPU buffer on CUDA
template<typename T>
void Conv2dBwdWeightInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    args_t *args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src1 = interfaces[0]->get_ptr<T>();
    const T *src2 = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::conv2d_bwd_weight_inplace::cuda<T>(
        stream,
        args->src1_m,
        args->src1_n,
        args->src1_channels,
        args->batch,
        args->src2_m,
        args->src2_n,
        args->stride_m,
        args->stride_n,
        args->src2_channels,
        args->offset_m,
        args->offset_n,
        args->alpha,
        src1,
        src2,
        args->dst_m,
        args->dst_n,
        args->dilation_m,
        args->dilation_n,
        args->beta,
        dst
    );
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for conv2d_bwd_weight_inplace tasks
template<typename T>
uint32_t Conv2dBwdWeightInplace<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    args_t *args = reinterpret_cast<args_t *>(task->cl_arg);
    // Apply hash over entire args
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(args, sizeof(*args), hash);
    return hash;
}

//! Submit conv2d_bwd_weight_inplace task
template<typename T>
void Conv2dBwdWeightInplace<std::tuple<T>>::submit(
    Index src1_m,
    Index src1_n,
    Index src1_channels,
    Index batch,
    Index src2_m,
    Index src2_n,
    Index stride_m,
    Index stride_n,
    Index src2_channels,
    Index offset_m,
    Index offset_n,
    Scalar alpha,
    Handle src1,
    Handle src2,
    Index dst_m,
    Index dst_n,
    Index dilation_m,
    Index dilation_n,
    Scalar beta,
    Handle dst
)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->src1_m = src1_m;
    args->src1_n = src1_n;
    args->src1_channels = src1_channels;
    args->batch = batch;
    args->src2_m = src2_m;
    args->src2_n = src2_n;
    args->stride_m = stride_m;
    args->stride_n = stride_n;
    args->src2_channels = src2_channels;
    args->offset_m = offset_m;
    args->offset_n = offset_n;
    args->alpha = alpha;
    args->dst_m = dst_m;
    args->dst_n = dst_n;
    args->dilation_m = dilation_m;
    args->dilation_n = dilation_n;
    args->beta = beta;
    enum starpu_data_access_mode dst_mode = STARPU_RW;
    if(beta == 0.0)
    {
        dst_mode = STARPU_W;
    }
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, src1.get(),
            STARPU_R, src2.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            dst_mode, dst.get(),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error(
                "Error in conv2d_bwd_weight_inplace task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Conv2dBwdWeightInplace<std::tuple<nntile::fp64_t>>;
template class Conv2dBwdWeightInplace<std::tuple<nntile::fp32_t>>;
template class Conv2dBwdWeightInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Conv2dBwdWeightInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Conv2dBwdWeightInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Conv2dBwdWeightInplace<std::tuple<nntile::bf16_t>>;

//! Pack of conv2d_bwd_weight_inplace operations for different types
conv2d_bwd_weight_inplace_pack_t conv2d_bwd_weight_inplace;

} // namespace nntile::starpu
