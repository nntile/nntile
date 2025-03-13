#include "nntile/tile/lars_tiled_step.hh"
#include "nntile/starpu/lars_tiled_step.hh"
#include <cmath>
#include <vector>

namespace nntile::kernel::lars_tiled_step {

template<typename T>
void lars_tiled_step(Index num_iter, Index num_elems, Index num_steps, Scalar gamma_0, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
                           const T *grad, T *momentum_buffer, T *weights) noexcept
{
    using Y = typename T::repr_t;
    const Y beta = weight_decay;
    
    // Compute the learning rate adjustment
    Y gamma_t = gamma_0 * std::pow(1 - static_cast<Y>(num_iter) / static_cast<Y>(num_steps), 2);
        
    // Cycle over the parameters (num_elems)
    for(Index i = 0; i < num_elems; ++i)
        {
            // Obtain the stochastic gradient for the current mini-batch
            Y grad_val = static_cast<Y>(grad[i]);
           
            // Compute local learning rate
            Y local_lr = std::sqrt(std::norm(static_cast<Y>(weights[i]))) / 
                         (std::norm(grad_val) + beta * std::norm(static_cast<Y>(weights[i])));
            
            // Update the momentum
            momentum_buffer[i] = momentum * static_cast<Y>(momentum_buffer[i]) + 
                                 gamma_t * local_lr * (grad_val + beta * static_cast<Y>(weights[i]));
            
            // Update the weights
            weights[i] = weights[i] - momentum_buffer[i];
        }
    
}

// Explicit instantiation
template
void lars_tiled_step<fp32_t>(Index num_iter, Index num_elems, Index num_steps, Scalar gamma_0, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
                                    const fp32_t *grad, fp32_t *momentum_buffer, fp32_t *weights) noexcept;

} // namespace nntile::kernel::lars_tiled_step
