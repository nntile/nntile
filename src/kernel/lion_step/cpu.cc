/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/lion_step/cpu.cc
 * Fused Lion step on buffers on CPU
 *
 * @version 1.1.0
 * */

 #include "nntile/kernel/lion_step/cpu.hh"
 #include <cmath>
 #include "nntile/kernel/cpu.hh"
 
 namespace nntile::kernel::lion_step
 {
 
 template<typename T>
 void cpu(Index num_iter, Index num_elems, Scalar beta_1_, Scalar beta_2_,
          Scalar lr_, Scalar weight_decay_, const T *grad,
         T *first_moment, T *p)
     noexcept
 //! Fused Lion step on buffers
 /*!
  *
  * @param[in] num_iter: current iteration number
  * @param[in] num_elems: Number of elements in buffers
  * @param[in] beta_1_: parameter for moving average of first moments
  * @param[in] beta_2_: parameter for moving average of second moments
  * @param[in] lr_: learning rate
  * @param[in] weight_decay_: coefficient for l2 regularizer
  * @param[in] grad_: Input buffer stored gradient, can be updated if weight_decay > 0
  * @param[inout] first_moment_: Input buffer stored first moments
  * @param[inout] p_: Input buffers with parameter that are updated in the end
  * */
 {
     using Y = typename T::repr_t;
     const Y beta_1{beta_1_}, beta_2{beta_2_}, lr{lr_},
           weight_decay{weight_decay_};
 
     // Lion-specific coefficients
     const Y beta1_complement = Y{1} - beta_1;
     const Y beta2_complement = Y{1} - beta_2;
 
     for(Index i = 0; i < num_elems; ++i)
     {
         Y p_val = static_cast<Y>(p[i]);
         const Y grad_val = static_cast<Y>(grad[i]);
 
         // Apply weight decay (same as AdamW)
         if (weight_decay != Y{0})
         {
             p_val *= Y{1} - lr * weight_decay;
         }
 
         // Update momentum buffer (first_moment)
         Y m_t;
         if(num_iter == 1)
         {
             m_t = beta1_complement * grad_val;
         }
         else
         {
             const Y m_prev = static_cast<Y>(first_moment[i]);
             m_t = beta_1 * m_prev + beta1_complement * grad_val;
         }
         first_moment[i] = static_cast<T>(m_t);
 
         // Compute Lion update direction
         const Y update_dir = beta_2 * m_t + beta2_complement * grad_val;
         const Y sign_update = (update_dir > Y{0}) ? Y{1} : 
                             ((update_dir < Y{0}) ? Y{-1} : Y{0});
 
         // Apply parameter update
         p[i] = static_cast<T>(p_val - lr * sign_update);
     }
 }
 
 // Explicit instantiation
 template
 void cpu<fp32_t>(Index num_iter, Index num_elems, Scalar beta_1, Scalar beta_2,
         Scalar lr, Scalar weight_decay, const fp32_t *grad,
         fp32_t *first_moment, fp32_t *p)
     noexcept;
 
 template
 void cpu<fp64_t>(Index num_iter, Index num_elems, Scalar beta_1, Scalar beta_2,
         Scalar lr, Scalar weight_decay, const fp64_t *grad,
         fp64_t *first_moment, fp64_t *p)
     noexcept;
 
 template
 void cpu<bf16_t>(Index num_iter, Index num_elems, Scalar beta_1, Scalar beta_2,
         Scalar lr, Scalar weight_decay, const bf16_t *grad,
         bf16_t *first_moment, bf16_t *p)
     noexcept;
 
 } // namespace nntile::kernel::lion_step
 
