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
 * @version 1.1.0
 * */

#include "nntile/kernel/subtract_indexed_outputs/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::subtract_indexed_outputs
{

template<typename T>
void cpu(Index n_labels, Index n_outputs, Scalar val_, const int64_t* labels_,
        T *dst_)
    noexcept
//! Subtraction of given val from indexed output of dst
/*! Mnemonically, the following operations are performed:
 *      dst[labels[i], i] -= val
 * for every i in [0, n_outputs)
 *
 * @param[in] n_labels: Number of possible labels
 * @param[in] n_outputs: Number of matrix elemets to update
 * @param[in] val_: Value that is subtracted from the matrix elements
 * @param[in] labels_: Index array of size n_outputs
 * @param[inout] dst_: Matrix of size n_labels by n_outputs continuously stored
 *      in Fortran order
 * */
{
    // using Y = typename CPUComputeType<T>::value;
    // auto dst = reinterpret_cast<Y *>(dst_);
    // const Y val{val_};
    using Y = typename T::repr_t;
    const Y val = static_cast<Y>(val_);
    using I = typename CPUComputeType<int64_t>::value;
    auto labels = reinterpret_cast<const I *>(labels_);
    Y dst_val{0.0};
    for(Index i = 0; i < n_outputs; ++i)
    {
        dst_val = static_cast<Y>(dst_[labels[i] + i*n_labels]);
        dst_[labels[i] + i*n_labels] = static_cast<T>(dst_val - val);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index n_labels, Index n_outputs, Scalar val,
        const int64_t* labels, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index n_labels, Index n_outputs, Scalar val,
        const int64_t* labels, fp64_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index n_labels, Index n_outputs, Scalar val,
        const int64_t* labels, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::subtract_indexed_outputs
