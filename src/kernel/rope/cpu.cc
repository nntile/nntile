/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/rope/cpu.cc
 * Rotary Positional Embedding
 *
 * @version 1.0.0
 * @author Gleb Karpov
 * @date 2024-06-27
 * */

#include "nntile/kernel/rope/cpu.hh"

namespace nntile
{
namespace kernel
{
namespace rope
{

template<typename T>
void cpu(Index m, Index k, Index l, const T *sin, const T *cos, 
        const T *src, T *dst)
    noexcept

/*! Change provided 2m-by-l src tensor and write result into dst tensor
 *  sin, cos m-by-k tensors. Each column holds sines and cosines.   
 *  dst[2i,j] = cos[i,j] * src[2i,j] - sin[i,j] * src[2i+1,j]
 *  dst[2i+1,j] = sin[i,j] * src[2i,j] + cos[i,j] * src[2i+1,j]
 *  In the application: 2m = head_size, k = n_seq x n_batch,
 *  l = n_seq x n_batch x n_head
 * @param[in] m: Size of the first mode of sin and cos tensors
 * @param[in] k: Size of the second mode of sin and cos tensors
 * @param[in] l: Size of the second mode of src and dst tensor
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] src: Input embedding tensor
 * @param[out] dst: Output embedding tensor with applied RoPE
 * */
{
    Index one = 1;
    Index two = 2;
    Index mk = m * k;
    Index h = l / k;
    
    // Use these angles for pairwise rotation of the same elements across all batches
    for (Index j = 0; j < h; ++j) {
        // Cycle over whole m*k elements of sin and cos buffers.
        for(Index i = 0; i < mk; ++i) {
            dst[two * (i + j*mk)] = cos[i] * src[two * (i + j*mk)] - sin[i] * src[two * (i + j*mk) + one];
            dst[two * (i + j*mk) + one] = sin[i] * src[two * (i + j*mk)] + cos[i] * src[two * (i + j*mk) + one];
        }
    }
     
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index k, Index l, const fp32_t *sin, const fp32_t *cos,
        const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index k, Index l, const fp64_t *sin, const fp64_t *cos, 
        const fp64_t *src, fp64_t *dst)
    noexcept;

} // namespace rope
} // namespace kernel
} // namespace nntile