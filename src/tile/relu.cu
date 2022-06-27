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
__global__ static
void cuda_relu_work(Index nelems, T *data)
{
    int start = threadIdx.x + blockIdx.x*blockDim.x,
        step = blockDim.x*gridDim.x;
    constexpr T zero{0};
    for(Index i = start; i < nelems; i += step)
    {
        if(data[i] < zero)
        {
            data[i] = zero;
        }
    }
}

template<typename T>
void relu_codelet_cuda(void *buffers[], void *cl_args)
{
    Index nelems;
    starpu_codelet_unpack_args(cl_args, &nelems);
    T *data = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    cudaStream_t stream = starpu_cuda_get_local_stream();
    dim3 grid(256), block(32);
    (cuda_relu_work<T>)<<<grid, block, 0, stream>>>(nelems, data);
}

template
void relu_codelet_cuda<fp32_t>(void *buffers[], void *cl_args);

template
void relu_codelet_cuda<fp64_t>(void *buffers[], void *cl_args);

} // namespace nntile

