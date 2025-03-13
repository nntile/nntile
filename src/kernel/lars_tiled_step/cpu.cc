#include "nntile/tile/lars_tiled_step.hh"
#include "nntile/starpu/lars_tiled_step.hh"
#include <cmath>
#include <vector>
#include <cblas.h> // Include CBLAS header for using cblas_dnrm2

namespace nntile::kernel::lars_step {

template<typename T>
void cpu(Index num_iter, Index num_elems, Index num_steps, Scalar gamma_0, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
               const T *grad, T *momentum_buffer, T *weights) noexcept
{
    using Y = typename T::repr_t;
    const Y beta = weight_decay;
    
    // Compute the learning rate adjustment
    Y gamma_t = gamma_0 * std::pow(1 - static_cast<Y>(num_iter) / static_cast<Y>(num_steps), 2);

    Y norm_grad = 0.0;
    Y norm_weights = 0.0;

    for(Index i = 0; i < num_elems; ++i)
    {
        // Obtain the stochastic gradient for the current mini-batch
        Y p_val=static_cast<Y>(weights[i]), grad_val = static_cast<Y>(grad[i]);
        
        // Update the momentum
        norm_grad += grad_val*grad_val
        norm_weights += p_val*p_val
    }

    norm_weights = std::sqrt(norm_weights);  // L2 norm of weights
    norm_grad = std::sqrt(norm_grad);       // L2 norm of grad

    // Compute local learning rate
    Y local_lr = lars_coefficient * norm_weights / 
                     (norm_grad + beta * norm_weights);
        
    // Cycle over the parameters (num_elems)
    for(Index i = 0; i < num_elems; ++i)
    {
        // Obtain the stochastic gradient for the current mini-batch
        Y p_val=static_cast<Y>(weights[i]), grad_val = static_cast<Y>(grad[i]);
        
        // Update the momentum
        Y momentum_buffer_val = momentum * static_cast<Y>(momentum_buffer[i]) + 
                             gamma_t * local_lr * (grad_val + beta * static_cast<Y>(weights[i]));
        momentum_buffer[i] = static_cast<T>(momentum_buffer_val);

        // Update the weights
        weights[i] = static_cast<T>(p_val - momentum_buffer_val);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index num_iter, Index num_elems, Index num_steps, Scalar gamma_0, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
                                    const fp32_t *grad, fp32_t *momentum_buffer, fp32_t *weights) noexcept;

template
void cpu<fp64_t>(Index num_iter, Index num_elems, Index num_steps, Scalar gamma_0, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
                                    const fp64_t *grad, fp64_t *momentum_buffer, fp64_t *weights) noexcept;

template
void cpu<bf16_t>(Index num_iter, Index num_elems, Index num_steps, Scalar gamma_0, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
                                    const bf16_t *grad, bf16_t *momentum_buffer, bf16_t *weights) noexcept;


} // namespace nntile::kernel::lars_tiled_step
