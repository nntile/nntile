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

namespace nntile::kernel::total_sum_accum
{

template<typename T>
void cpu(T alpha, Index n_labels, Index n_outputs, const T* logsumexp, const T* src,
        const Index* labels, T *val)
    noexcept
//! Total sum accumulating from logsumexp and corrected by elements from src
/*! Mnemonically, the following operations are performed:
 * for every i in [0, n_outputs)
 *      val += alpha * (logsumexp[i]-src[labels[i], i]);
 *
 * @param[in] alpha: Scalar multiplier
 * @param[in] n_labels: Number of possible labels
 * @param[in] n_outputs: Number of elements to sum up.
 * @param[in] logsumexp: Array with logsumexp values of size n_outputs.
 * @param[in] src: Matrix of size n_labels times n_outputs stored continuously
 *      in Fortran order
 * @param[in] labels: Array of size n_outputs with correct labels
 * @param[inout] val: Scalar that accumulates the total sum
 * */
{
    constexpr T zero = 0.0;
    T sum = zero, c = zero, y, t;
    for(Index i = 0; i < n_outputs; ++i)
    {
        //*val += logsumexp[i] - src[labels[i] + i*n_labels];
        y = logsumexp[i] - c;
        t = sum + y;
        c = (t-sum) - y;
        sum = t;
        y = - src[labels[i] + i*n_labels] - c;
        t = sum + y;
        c = (t-sum) - y;
        sum = t;
    }
    *val = (*val-alpha*c) + alpha*sum;
    //std::cout << "loss=" << *val << "\n";
}

// Explicit instantiation
template
void cpu<fp32_t>(fp32_t alpha, Index n_labels, Index n_outputs, const fp32_t* logsumexp,
        const fp32_t* src, const Index* class_labels, fp32_t* val)
    noexcept;

template
void cpu<fp64_t>(fp64_t alpha, Index n_labels, Index n_outputs, const fp64_t* logsumexp,
        const fp64_t* src, const Index* class_labels, fp64_t *val)
    noexcept;

} // namespace nntile::kernel::total_sum_accum
