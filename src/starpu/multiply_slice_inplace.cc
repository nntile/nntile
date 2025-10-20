/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/multiply_slice_inplace.cc
 * StarPU wrappers for in-place multiplication of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/multiply_slice_inplace.hh"
#include "nntile/starpu/config.hh"

#include "nntile/kernel/multiply_slice_inplace.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
MultiplySliceInplace<std::tuple<T>>::MultiplySliceInplace():
    codelet("nntile_multiply_slice_inplace", footprint, cpu_funcs, cuda_funcs)
{}

//! StarPU wrapper for kernel::multiply_slice_inplace::cpu<T>
template<typename T>
void MultiplySliceInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    auto args = reinterpret_cast<args_t *>(cl_args);
    auto m = args->m;
    auto n = args->n;
    auto k = args->k;
    auto alpha = args->alpha;
    auto src = reinterpret_cast<const T *>(STARPU_VECTOR_GET_PTR(buffers[0]));
    auto dst = reinterpret_cast<T *>(STARPU_VECTOR_GET_PTR(buffers[1]));
    kernel::multiply_slice_inplace::cpu<T>(m, n, k, alpha, src, dst);
}

//! StarPU wrapper for kernel::multiply_slice_inplace::cuda<T>
template<typename T>
void MultiplySliceInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifdef NNTILE_USE_CUDA
    auto args = reinterpret_cast<args_t *>(cl_args);
    auto m = args->m;
    auto n = args->n;
    auto k = args->k;
    auto alpha = args->alpha;
    auto src = reinterpret_cast<const T *>(STARPU_VECTOR_GET_PTR(buffers[0]));
    auto dst = reinterpret_cast<T *>(STARPU_VECTOR_GET_PTR(buffers[1]));
    auto stream = starpu_cuda_get_local_stream();

    kernel::multiply_slice_inplace::cuda<T>(stream, m, n, k, alpha, src, dst);
#else // NNTILE_USE_CUDA
    // CUDA not available
    (void)buffers;
    (void)cl_args;
#endif // NNTILE_USE_CUDA
}

//! Footprint for multiply_slice_inplace tasks
template<typename T>
uint32_t MultiplySliceInplace<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t*>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    return hash;
}

//! Submit multiply_slice_inplace task into StarPU pool of tasks
template<typename T>
void MultiplySliceInplace<std::tuple<T>>::submit(Index m, Index n, Index k,
        Scalar alpha, Handle src, Handle dst, Index axis)
{
    // Codelet arguments
    args_t args{m, n, k, alpha, axis};

    // Submit task
    int ret = starpu_task_insert(&codelet,
                                 STARPU_VALUE, &args, sizeof(args),
                                 STARPU_R, src,
                                 STARPU_RW, dst,
                                 0);
    if(ret != 0)
    {
        throw std::runtime_error("Error in multiply_slice_inplace task submission");
    }
}

// Explicit instantiation for all supported types
template class MultiplySliceInplace<std::tuple<fp64_t>>;
template class MultiplySliceInplace<std::tuple<fp32_t>>;
template class MultiplySliceInplace<std::tuple<fp32_fast_tf32_t>>;
template class MultiplySliceInplace<std::tuple<fp32_fast_fp16_t>>;
template class MultiplySliceInplace<std::tuple<fp32_fast_bf16_t>>;
template class MultiplySliceInplace<std::tuple<bf16_t>>;
template class MultiplySliceInplace<std::tuple<fp16_t>>;


//! Pack of multiply_slice_inplace operations for different types
multiply_slice_inplace_pack_t multiply_slice_inplace;

} // namespace nntile::starpu
