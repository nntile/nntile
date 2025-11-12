/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/accumulate_attn_output/cpu.cc
 * Accumulate attention outputs on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/accumulate_attn_output/cpu.hh"

#include <algorithm>
#include <cmath>
#include <limits>

#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::accumulate_attn_output
{

namespace
{

using lse_repr_t = typename fp32_t::repr_t;

inline bool is_neg_inf(lse_repr_t value)
{
    return std::isinf(value) && value < lse_repr_t(0);
}

} // namespace

template<typename T>
void cpu(Index seq, Index batch, const fp32_t *src_lse, const T *src_attn,
        fp32_t *dst_lse, T *dst_attn)
    noexcept
//! Accumulate attention outputs on CPU
/*! For every (sequence, batch) pair this kernel merges source statistics
 *  and values into the destination buffers using numerically stable
 *  log-sum-exp coefficients.
 *
 *  @param[in] seq: Sequence length (number of rows)
 *  @param[in] batch: Combined batch size (number of columns)
 *  @param[in] src_lse: Source log-sum-exp statistics [seq, batch]
 *  @param[in] src_attn: Source attention output [seq, batch]
 *  @param[in,out] dst_lse: Destination log-sum-exp statistics [seq, batch]
 *  @param[in,out] dst_attn: Destination attention output [seq, batch]
 * */
{
    if(seq <= 0 || batch <= 0)
    {
        return;
    }

    for(Index s = 0; s < seq; ++s)
    {
        const Index row_offset = s * batch;
        for(Index b = 0; b < batch; ++b)
        {
            const Index idx = row_offset + b;
            const lse_repr_t old_lse =
                    static_cast<lse_repr_t>(dst_lse[idx]);
            const lse_repr_t incoming_lse =
                    static_cast<lse_repr_t>(src_lse[idx]);

            const bool dst_active = !is_neg_inf(old_lse);
            const bool src_active = !is_neg_inf(incoming_lse);

            if(!dst_active && !src_active)
            {
                continue;
            }

            lse_repr_t new_lse = old_lse;
            if(dst_active && src_active)
            {
                const lse_repr_t max_lse = std::max(old_lse, incoming_lse);
                const lse_repr_t sum = std::exp(old_lse - max_lse)
                        + std::exp(incoming_lse - max_lse);
                new_lse = max_lse + std::log(sum);
            }
            else if(src_active)
            {
                new_lse = incoming_lse;
            }

            const lse_repr_t dst_weight =
                    dst_active ? std::exp(old_lse - new_lse) : lse_repr_t(0);
            const lse_repr_t src_weight =
                    src_active ? std::exp(incoming_lse - new_lse) : lse_repr_t(0);

            using repr_t = typename T::repr_t;
            const repr_t dst_val = static_cast<repr_t>(dst_attn[idx]);
            const repr_t src_val = static_cast<repr_t>(src_attn[idx]);
            const repr_t updated =
                    static_cast<repr_t>(dst_weight) * dst_val
                    + static_cast<repr_t>(src_weight) * src_val;

            dst_lse[idx] = new_lse;
            dst_attn[idx] = static_cast<T>(updated);
        }
    }
}

// Explicit instantiation
template
void cpu<fp16_t>(Index seq, Index batch, const fp32_t *src_lse,
        const fp16_t *src_attn, fp32_t *dst_lse, fp16_t *dst_attn)
    noexcept;

template
void cpu<bf16_t>(Index seq, Index batch, const fp32_t *src_lse,
        const bf16_t *src_attn, fp32_t *dst_lse, bf16_t *dst_attn)
    noexcept;

template
void cpu<fp32_t>(Index seq, Index batch, const fp32_t *src_lse,
        const fp32_t *src_attn, fp32_t *dst_lse, fp32_t *dst_attn)
    noexcept;

template
void cpu<fp64_t>(Index seq, Index batch, const fp32_t *src_lse,
        const fp64_t *src_attn, fp32_t *dst_lse, fp64_t *dst_attn)
    noexcept;

} // namespace nntile::kernel::accumulate_attn_output
