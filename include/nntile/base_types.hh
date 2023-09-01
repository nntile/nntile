/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/base_types.hh
 * Base integer and floating point types.
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-06-29
 * */

#pragma once

#include <cstdint>

namespace nntile
{

//! Large enough signed integer for indexing purposes
using Index = int64_t;

// Supported computational types
//! Double precision alias
using fp64_t = double;
//! Single precision alias
using fp32_t = float;
//! Half precision FP16 mock type
class fp16_t
{
private:
    int16_t _;
};

// Boolean type for mask
using bool_t = bool;

// Add more types like fp16_t, bf16_t and tf32_t in the future

} // namespace nntile

