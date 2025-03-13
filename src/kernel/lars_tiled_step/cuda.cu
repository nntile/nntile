//Not very correct CUDA implementation

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
        Y w_val = Y{weights[i]}, grad_val = Y{grad[i]};

        // NOT CORRECT Update local learning rates and momentums
        Y local_lr = eta * std::sqrt(std::norm(static_cast<Y>(weights[i]))) / 
                     (std::norm(grad_val) + beta * std::norm(static_cast<Y>(weights[i])));

        // PROBABLY NOT OPTIMAL
        Y gamma_t = gamma_0 * (1 - static_cast<Y>(num_iter) / static_cast<Y>(num_steps));
        momentum_buffer_val = momentum * static_cast<Y>(momentum_buffer[i]) + gamma_t * local_lr * (grad_val + weight_decay * w_val);
        w[i] -= momentum_buffer_val;        
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

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index num_iter, Index num_elems, Index num_steps, Scalar gamma_0,
          Scalar momentum, Scalar weight_decay, Scalar eta, const fp32_t *gradients, 
          fp32_t *momentum_buffer, fp32_t *weights)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index num_iter, Index num_elems, Index num_steps, Scalar gamma_0,
          Scalar momentum, Scalar weight_decay, Scalar eta, const fp64_t *gradients, 
          fp64_t *momentum_buffer, fp64_t *weights)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index num_iter, Index num_elems, Index num_steps, Scalar gamma_0,
          Scalar momentum, Scalar weight_decay, Scalar eta, const bf16_t *gradients, 
          bf16_t *momentum_buffer, bf16_t *weights)
    noexcept;

} // namespace nntile::kernel::lars_tiled_step
