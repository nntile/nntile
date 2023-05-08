/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/total_sum_accum/cpu.cc
 * Total sum accumulated of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-15
 * */

#include "nntile/kernel/total_sum_accum/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace total_sum_accum
{

template<typename T>
void cpu(Index n_row, const T* logsumexp, const T* src, const Index* class_labels, T *val)
    noexcept
//! Total sum accimulating from logsumexp and corrected by elements from src
/*! Mnemonically, the following operations are performed:
 *      val += logsumexp[i] - src[i, class_labels[i]];
 * for every i in [0, n_row)
 *
 * @param[in] n_row: Size of the class_labels and numner of rows stored in dst.
 * @param[in] logsumexp: Array with logsumexp values of size n_row
 * @param[in] src: Matrix of size n_row times n_classes stored continuously in Fortran order
 * @param[in] class_labels: Array of size n_row with indices of columns of matrix stored in src
 * @param[inout] val: Scalar that accumulates the total sum
 * */
{
    for (Index i = 0; i < n_row; ++i)
    {
        *val += logsumexp[i] - src[n_row * class_labels[i] + i];
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index n_row, const fp32_t* logsumexp,
                 const fp32_t* src, const Index* class_labels, fp32_t* val)
    noexcept;

template
void cpu<fp64_t>(Index n_row, const fp64_t* logsumexp,
                 const fp64_t* src, const Index* class_labels, fp64_t *val)
    noexcept;

} // namespace total_sum_accum
} // namespace kernel
} // namespace nntile