/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/scale_slice.cc
 * StarPU wrappers for scaling of a broadcasted slice
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/scale_slice.hh"
#include "nntile/kernel/scale_slice.hh"
#include <starpu.h>

namespace nntile::starpu
{

//! Wrapper for a generic CPU implementation
template<typename T>
void ScaleSlice<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface *>(buffers);
    const T *src = interfaces[0].get_ptr<T>();
    T *dst = interfaces[1].get_ptr<T>();
    // Launch kernel
    kernel::scale_slice::cpu<T>(args->m, args->n, args->k, args->alpha, src, dst);
}

#ifdef NNTILE_USE_CUDA
//! Wrapper for a generic CUDA implementation
template<typename T>
void ScaleSlice<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface *>(buffers);
    const T *src = interfaces[0].get_ptr<T>();
    T *dst = interfaces[1].get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::scale_slice::cuda<T>(stream, args->m, args->n, args->k, args->alpha, src, dst);
}
#endif // NNTILE_USE_CUDA

//! Footprint function for the current operation
template<typename T>
uint32_t ScaleSlice<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(starpu_task_get_cl_arg(task));
    // Apply hash over parameters m, n, k, alpha
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    hash = starpu_hash_crc32c_be_n(&args->alpha, sizeof(args->alpha), hash);
    return hash;
}

//! Constructor
template<typename T>
ScaleSlice<std::tuple<T>>::ScaleSlice()
{
    // Create codelet
    codelet = CodeletTyped<T>(
        "scale_slice",
        footprint,
        {cpu_funcs.begin(), cpu_funcs.end()},
        {cuda_funcs.begin(), cuda_funcs.end()},
        [](struct starpu_task *task) -> int
        {
            // Get arguments
            auto args = reinterpret_cast<args_t *>(starpu_task_get_cl_arg(task));
            // Check arguments
            if(args->m == 0)
            {
                return -EINVAL;
            }
            if(args->n == 0)
            {
                return -EINVAL;
            }
            if(args->k == 0)
            {
                return -EINVAL;
            }
            // Get interfaces
            auto interfaces = reinterpret_cast<VariableInterface *>(starpu_task_get_interfaces(task));
            // Check interfaces
            if(interfaces[0].get_nbytes() != args->m * args->n * sizeof(T))
            {
                return -EINVAL;
            }
            if(interfaces[1].get_nbytes() != args->m * args->k * args->n * sizeof(T))
            {
                return -EINVAL;
            }
            return 0;
        }
    );
}

//! Submit scale_slice task
template<typename T>
void ScaleSlice<std::tuple<T>>::submit(
        Index m,
        Index n,
        Index k,
        Scalar alpha,
        Handle src,
        Handle dst
    )
{
    // Access mode for the source handle
    constexpr unsigned src_mode = STARPU_R;
    // Access mode for the destination handle
    constexpr unsigned dst_mode = STARPU_W;
    // Submit task
    int ret = starpu_task_submit(
        starpu_task_create()
        .tag(STARPU_ASYNC)
        .cl(&codelet, sizeof(codelet))
        .cl_arg(&args_t{m, n, k, alpha}, sizeof(args_t))
        .cl_arg(src, src_mode)
        .cl_arg(dst, dst_mode)
    );
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in starpu_task_submit");
    }
}

// Explicit instantiation
template class ScaleSlice<std::tuple<fp64_t>>;
template class ScaleSlice<std::tuple<fp32_t>>;
template class ScaleSlice<std::tuple<fp32_fast_tf32_t>>;
template class ScaleSlice<std::tuple<fp32_fast_fp16_t>>;
template class ScaleSlice<std::tuple<fp32_fast_bf16_t>>;
template class ScaleSlice<std::tuple<bf16_t>>;
template class ScaleSlice<std::tuple<fp16_t>>;

//! Pack of scale_slice operations for different types
scale_slice_pack_t scale_slice;

} // namespace nntile::starpu
