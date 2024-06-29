/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/base_types.cc
 * Base integer and floating point types.
 *
 * @version 1.0.0
 * */

#include "nntile/base_types.hh"
#include <type_traits>
#include "nntile/defs.h"

#ifdef NNTILE_USE_CUDA
#   include <cuda_bf16.h>
#endif

namespace nntile
{

//! Shorthand for nntile::bf16_t internal type
using bf16_i_t = typename bf16_t::internal_t;

//! Shorthand for nntile::bf16_t compat type
using bf16_c_t = typename bf16_t::compat_t;

//! Conversion from compat to internal for nntile::bf16_t
bf16_i_t bf16_c_to_i(const bf16_c_t &value)
{
    // Check that compat type is float
    static_assert(std::is_same<bf16_c_t, float>::value);
    // Convert compat (float) value to 16-bit BrainFloat16
    auto actual = __float2bfloat16(value);
    // Convert without touching bits
    return *reinterpret_cast<bf16_i_t *>(&actual);
}

// BF16: Conversion from internal to compat for nntile::bf16_t
bf16_c_t bf16_i_to_c(const bf16_i_t &value)
{
    // Check that compat type is float
    static_assert(std::is_same<bf16_c_t, float>::value);
    // Convert without touching bits
    auto actual = *reinterpret_cast<const __nv_bfloat16 *>(&value);
    // Convert 16-bit BrainFloat16 value to compat
    return __bfloat162float(actual);
}

// BF16: Constructor from a compat_t value
bf16_t::bf16_t(const bf16_c_t &other):
    value(bf16_c_to_i(other))
{
}

// BF16: Assignment from a compat_t value
bf16_t &bf16_t::operator=(const bf16_c_t &other)
{
    value = bf16_c_to_i(other);
    return *this;
}

// BF16: Conversion to compat_t value
bf16_t::operator compat_t() const
{
    return bf16_i_to_c(value);
}

} // namespace nntile
