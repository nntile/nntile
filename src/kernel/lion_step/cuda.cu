#include "nntile/kernel/lion_step/cuda.hh"
#include "nntile/kernel/cuda.hh" 
#include <cstdio>
#include <cuda_runtime.h>

namespace nntile::kernel::lion_step {

template<typename T>
static __global__
void cuda_kernel(Index num_iter, Index num_elems,
                   typename T::repr_t beta_1,
                   typename T::repr_t beta_2,
                   typename T::repr_t lr,
                   typename T::repr_t weight_decay,
                   const T *grad,
                   T *first_moment,
                   T *p)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    using Y = typename T::repr_t;
    if (i < num_elems)
    {
        // Reading parameter and gradient values
        Y p_val = Y{p[i]};
        Y grad_val = Y{grad[i]};

        // Apply weight decay (similar to AdamW)
        if (weight_decay != Y{0})
        {
            p_val *= (Y{1} - lr * weight_decay);
        }

        // Update the first momentum buffer
        Y m_t;
        Y beta1_complement = Y{1} - beta_1;
        if (num_iter == 1)
        {
            m_t = beta1_complement * grad_val;
        }
        else
        {
            Y m_prev = Y{first_moment[i]};
            m_t = beta_1 * m_prev + beta1_complement * grad_val;
        }
        first_moment[i] = static_cast<T>(m_t);

        // Calculate the update direction (Lion update direction)
        Y beta2_complement = Y{1} - beta_2;
        Y update_dir = beta_2 * m_t + beta2_complement * grad_val;
        Y sign_update = (update_dir > Y{0}) ? Y{1} :
                        ((update_dir < Y{0}) ? Y{-1} : Y{0});

        // Updating the parameter
        p[i] = static_cast<T>(p_val - lr * sign_update);
    }
}

template<typename T>
void cuda(cudaStream_t stream,
          Index num_iter,
          Index num_elems,
          Scalar beta_1,
          Scalar beta_2,
          Scalar lr,
          Scalar weight_decay,
          const T *grad,
          T *first_moment,
          T *p)
    noexcept
{
    dim3 blocks((num_elems + 255) / 256), threads(256);
    using Y = typename T::repr_t;
    cuda_kernel<T><<<blocks, threads, 0, stream>>>(
        num_iter, num_elems,
        Y{beta_1}, Y{beta_2}, Y{lr}, Y{weight_decay},
        grad, first_moment, p
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
         printf("CUDA error in lion_step: %s\n", cudaGetErrorString(err));
    }
}

// Explicit instantiation of templates for the types used
template void cuda<fp32_t>(cudaStream_t stream,
                           Index num_iter,
                           Index num_elems,
                           Scalar beta_1,
                           Scalar beta_2,
                           Scalar lr,
                           Scalar weight_decay,
                           const fp32_t *grad,
                           fp32_t *first_moment,
                           fp32_t *p)
    noexcept;
template void cuda<fp64_t>(cudaStream_t stream,
                           Index num_iter,
                           Index num_elems,
                           Scalar beta_1,
                           Scalar beta_2,
                           Scalar lr,
                           Scalar weight_decay,
                           const fp64_t *grad,
                           fp64_t *first_moment,
                           fp64_t *p)
    noexcept;
template void cuda<bf16_t>(cudaStream_t stream,
                           Index num_iter,
                           Index num_elems,
                           Scalar beta_1,
                           Scalar beta_2,
                           Scalar lr,
                           Scalar weight_decay,
                           const bf16_t *grad,
                           bf16_t *first_moment,
                           bf16_t *p)
    noexcept;

} // namespace nntile::kernel::lion_step
