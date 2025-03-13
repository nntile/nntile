#include "nntile/kernel/lars_tiled_step/cuda.hh"
#include "nntile/kernel/cuda.hh"
#include <cmath>
#include <vector>

namespace nntile::kernel::lars_tiled_step
{

template<typename T>
static global
void cuda_kernel(Index num_iter, Index num_elems, Index num_steps, typename T::repr_t gamma_0, typename T::repr_t momentum, 
                 typename T::repr_t weight_decay, typename T::repr_t eta, 
                 T *weights, const T *gradients, T *momentum_buffer)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    using Y = typename T::repr_t;

    if (i < num_elems)
    {
        Y w_val = weights[i];
        Y g_val = gradients[i];

        // Update local learning rates and momentums
        Y local_lr = std::sqrt(std::norm(static_cast<Y>(weights[i]))) / 
                     (std::norm(grad_val) + beta * std::norm(static_cast<Y>(weights[i])));
        
        Y gamma_t = gamma_0 * (1 - static_cast<Y>(num_iter) / static_cast<Y>(num_steps));
        momentum_buffer[i] = momentum * momentum_buffer[i] + gamma_t * local_lr * (g_val + weight_decay * w_val);
        weights[i] -= momentum_buffer[i];        
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index num_iter, Index num_elems, Index num_steps, Scalar gamma_0,
          Scalar momentum, Scalar weight_decay, Scalar eta, const T *gradients, 
          T *momentum_buffer, T *weights) noexcept
{
    dim3 blocks((num_elems + 255) / 256);
    dim3 threads(256);
    
    // Launch the CUDA kernel
    cuda_kernel<T><<<blocks, threads, 0, stream>>>(num_iter, num_elems, num_steps, Y{gamma_0}, 
                Y{momentum}, Y{weight_decay}, Y{eta}, gradients, momentum_buffer, weights);
}

} // namespace nntile::kernel::lars_tiled_step
