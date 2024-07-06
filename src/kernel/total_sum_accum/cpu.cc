/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/total_sum_accum/cpu.cc
 * Total sum accumulated of a buffer on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/total_sum_accum/cpu.hh"
#include <cmath>
#include <iostream>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::total_sum_accum
{

template<typename T>
void cpu(Scalar alpha_, Index n_labels, Index n_outputs, const T* logsumexp_,
        const T* src_, const int64_t* labels_, T *val_)
    noexcept
//! Total sum accumulating from logsumexp and corrected by elements from src
/*! Mnemonically, the following operations are performed:
 * for every i in [0, n_outputs)
 *      val += alpha * (logsumexp[i]-src[labels[i], i]);
 *
 * @param[in] alpha_: Scalar multiplier
 * @param[in] n_labels: Number of possible labels
 * @param[in] n_outputs: Number of elements to sum up.
 * @param[in] logsumexp_: Array with logsumexp values of size n_outputs.
 * @param[in] src_: Matrix of size n_labels times n_outputs stored continuously
 *      in Fortran order
 * @param[in] labels_: Array of size n_outputs with correct labels
 * @param[inout] val_: Scalar that accumulates the total sum
 * */
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    constexpr Y zero{0.0};
    Y sum = zero, c = zero, y, t;
    using I = typename CPUComputeType<int64_t>::value;
    auto labels = reinterpret_cast<const I *>(labels_);
    Y logsumexp_val{0.0};
    Y src_val{0.0};
    for(Index i = 0; i < n_outputs; ++i)
    {
        //*val += logsumexp[i] - src[labels[i] + i*n_labels];
        logsumexp_val = static_cast<Y>(logsumexp_[i]);
        src_val = static_cast<Y>(src_[labels[i] + i*n_labels]);
        y = logsumexp_val - c;
        t = sum + y;
        c = (t-sum) - y;
        sum = t;
        y = -src_val - c;
        t = sum + y;
        c = (t-sum) - y;
        sum = t;
    }
    Y cur_val = static_cast<Y>(*val_);
    *val_ = static_cast<T>((cur_val-alpha*c) + alpha*sum);
    //std::cout << "loss=" << *val << "\n";
}

// Explicit instantiation
template
void cpu<fp32_t>(Scalar alpha, Index n_labels, Index n_outputs, const fp32_t* logsumexp,
        const fp32_t* src, const int64_t* labels, fp32_t* val)
    noexcept;

template
void cpu<fp64_t>(Scalar alpha, Index n_labels, Index n_outputs, const fp64_t* logsumexp,
        const fp64_t* src, const int64_t* labels, fp64_t *val)
    noexcept;

template
void cpu<bf16_t>(Scalar alpha, Index n_labels, Index n_outputs, const bf16_t* logsumexp,
        const bf16_t* src, const int64_t* labels, bf16_t *val)
    noexcept;

} // namespace nntile::kernel::total_sum_accum
