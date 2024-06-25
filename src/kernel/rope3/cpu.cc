/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/rope3/cpu.cc
 * ROtary Positional Embedding
 *
 * @version 1.0.0
 * @author Gleb Karpov
 * @date 2024-05-22
 * */

#include "nntile/kernel/rope3/cpu.hh"

namespace nntile
{
namespace kernel
{
namespace rope3
{

template<typename T>
void cpu(Index m, Index n, Index b, Index s, const T *sin, const T *cos, 
        const T *src, T *dst)
    noexcept

/*! Change provided m-by-n tensor embed
 *  sin, cos (n_head * head_size/2)-by-n_seq tensors. Each column holds sines and cosines for the specific token in a sequence.    
 *  src, dst: head_size-by-(n_seq * n_batch * n_head) tensors, 
 * @param[in] m: Size of the first mode of src tensor
 * @param[in] n: Size of the second mode of src tensor; n = b * s * n_head
 * @param[in] b: batch size
 * @param[in] s: n_seq, also the size of the second mode of sin and cos tensor
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] src: Input embedding tensor
 * @param[out] dst: Output embedding tensor with applied RoPE
 * */
{
    Index msb = m * s * b;
    Index tmp = 2;
    Index n_head = n / (s * b);
    Index half_hs = m / tmp;
    Index rot_index = 0;

    // Cycle over n_head
    for(Index i3 = 0; i3 < n_head; ++i3)
    {
        const T *sin_fiber = sin + i3 * half_hs;
        const T *cos_fiber = cos + i3 * half_hs;
        rot_index = 0;

        const T *src_slice = src + i3 * msb;
        T *dst_slice = dst + i3 * msb;

        // Cycle over whole slice m*k of output buffer. Pairwise rotation of elements.
        for(Index i1 = 0; i1 < msb-1; i1 += 2)
        {       
        
        dst_slice[i1] = src_slice[i1] * cos_fiber[rot_index] - src_slice[i1+1] * sin_fiber[rot_index];
        dst_slice[i1+1] = src_slice[i1] * sin_fiber[rot_index] + src_slice[i1+1] * cos_fiber[rot_index];

        rot_index += 1;

        if (rot_index == (half_hs-1)) {
            rot_index = 0;
            *sin_fiber += half_hs * n_head;
            *cos_fiber += half_hs * n_head;
        }

        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index b, Index s, const fp32_t *sin, const fp32_t *cos, 
        const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index b, Index s, const fp64_t *sin, const fp64_t *cos,
        const fp64_t *src, fp64_t *dst)
    noexcept;

} // namespace rope3
} // namespace kernel
} // namespace nntile