/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/subtract_indexed_column/cpu.cc
 * Subtract a value from the indexed column of matrix in a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-18
 * */

#include "nntile/kernel/subtract_indexed_column/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace subtract_indexed_column
{

template<typename T>
void cpu(Index n_row, T val, const Index* class_labels, T *dst)
    noexcept
//! Subtraction of given val from indexed column of dst
/*! Mnemonically, the following operations are performed:
 *      dst[i, class_labels[i]] -= val
 * for every i in [0, n_row)
 *
 * @param[in] n_row: Size of the class_labels and numner of rows stored in dst.
 * @param[in] val: Value that is subtracted from the matrix elements
 * @param[in] class_labels: Index array of size n_row, where indices of columns 
 * from the dst matrix are stored 
 * @param[inout] dst: Matrix of size n_row by n_classes continuously stored in Fortran order
 * which is updated inplace with subtraction val from columns indexed by class_labels and 
 * sequential row indices
 * */
{
    for (Index i = 0; i < n_row; ++i)
    {
        dst[n_row * class_labels[i] + i] -= val;
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index n_row, fp32_t val, const Index* class_labels, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index n_row, fp64_t val, const Index* class_labels, fp64_t *dst)
    noexcept;

} // namespace subtract_indexed_column
} // namespace kernel
} // namespace nntile