/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/mask_scalar/cpu.cc
 * Mask operation with scalar on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/mask_scalar/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::mask_scalar
{

template<typename T>
void cpu(Index nrows, Index ncols, const bool_t *mask_, Scalar val_, T *data)
    noexcept
//! Set certain matrix entries to a given value by mask on CPU
/*! Does the following operation:
 *      if(!mask[i]) data[i,:] = val
 *
 * @params[in] nrows: Number of rows of data
 * @params[in] ncols: Number of columns of data
 * @params[in] mask_: buffer with mask values with nrows entries
 * @params[in] val_: value to set if mask element is false
 * @params[inout] data: nrows by ncols matrix, whose elements are updated
 * */
{
    using Y = typename T::repr_t;
    using B = typename CPUComputeType<bool_t>::value;
    auto mask = reinterpret_cast<const B *>(mask_);
    const Y val{val_};
    for(Index i = 0; i < nrows; ++i)
    {
        if(!mask[i])
        {
            for(Index j = 0; j < ncols; ++j)
            {
                data[j*nrows+i] = static_cast<T>(val);
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nrows, Index ncols, const bool_t *mask, Scalar val,
        fp32_t *data)
    noexcept;

template
void cpu<fp64_t>(Index nrows, Index ncols, const bool_t *mask, Scalar val,
        fp64_t *data)
    noexcept;

template
void cpu<fp32_fast_tf32_t>(Index nrows, Index ncols, const bool_t *mask, Scalar val,
        fp32_fast_tf32_t *data)
    noexcept;

template
void cpu<bf16_t>(Index nrows, Index ncols, const bool_t *mask, Scalar val,
        bf16_t *data)
    noexcept;

} // namespace nntile::kernel::mask_scalar
