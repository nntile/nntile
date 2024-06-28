/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/subtract_indexed_outputs/cpu.cc
 * Subtract a value from certain elements of a matrix on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/subtract_indexed_outputs/cpu.hh"
#include <cmath>

namespace nntile::kernel::subtract_indexed_outputs
{

template<typename T>
void cpu(Index n_labels, Index n_outputs, T val, const Index* labels, T *dst)
    noexcept
//! Subtraction of given val from indexed output of dst
/*! Mnemonically, the following operations are performed:
 *      dst[labels[i], i] -= val
 * for every i in [0, n_outputs)
 *
 * @param[in] n_labels: Number of possible labels
 * @param[in] n_outputs: Number of matrix elemets to update
 * @param[in] val: Value that is subtracted from the matrix elements
 * @param[in] labels: Index array of size n_outputs
 * @param[inout] dst: Matrix of size n_labels by n_outputs continuously stored
 *      in Fortran order
 * */
{
    for(Index i = 0; i < n_outputs; ++i)
    {
        dst[labels[i] + i*n_labels] -= val;
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index n_labels, Index n_outputs, fp32_t val,
        const Index* labels, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index n_labels, Index n_outputs, fp64_t val,
        const Index* labels, fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::subtract_indexed_outputs
