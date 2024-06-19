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
 * @version 1.0.0
 * */

#include "nntile/kernel/mask_scalar/cpu.hh"

namespace nntile::kernel::mask_scalar
{

template<typename T>
void cpu(Index nrows, Index ncols, const bool_t *mask, T val, T *data)
    noexcept
//! Set certain matrix entries to a given value by mask on CPU
/*! Does the following operation:
 *      if(!mask[i]) data[i,:] = val
 *
 * @params[in] nrows: Number of rows of data
 * @params[in] ncols: Number of columns of data
 * @params[in] mask: buffer with mask values with nrows entries
 * @params[in] val: value to set if mask element is false
 * @params[in,out] data: nrows by ncols matrix, whose elements are updated
 * */
{
    for(Index i = 0; i < nrows; ++i)
    {
        if(!mask[i])
        {
            for(Index j = 0; j < ncols; ++j)
            {
                data[j*nrows+i] = val;
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nrows, Index ncols, const bool_t *mask, fp32_t val,
        fp32_t *data)
    noexcept;

template
void cpu<fp64_t>(Index nrows, Index ncols, const bool_t *mask, fp64_t val,
        fp64_t *data)
    noexcept;

} // namespace nntile::kernel::mask_scalar

