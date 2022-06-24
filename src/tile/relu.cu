/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/relu.cu
 * ReLU operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/relu.hh"

namespace nntile
{

template<typename T>
__global__ static void cuda_relu_work(T *data)
{
    int i = blockIdx.x;
    constexpr T zero{0};
    if(data[i] < zero)
    {
        data[i] = zero;
    }
}

template<typename T>
void relu_codelet_gpu(void *buffers[], void *cl_args)
{
    Index nelems;
    starpu_codelet_unpack_args(cl_args, &nelems);
    T *data = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    cudaStream_t stream = starpu_cuda_get_local_stream();
    dim3 threads(1);
    (cuda_relu_work<T>)<<<nelems, 1, 0, stream>>>(data);
}

template
void relu_codelet_gpu<fp32_t>(void *buffers[], void *cl_args);

template
void relu_codelet_gpu<fp64_t>(void *buffers[], void *cl_args);

} // namespace nntile

