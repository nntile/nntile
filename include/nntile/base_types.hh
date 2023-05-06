/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/base_type.hh
 * Base integer and floating point types.
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-04
 * */

#pragma once

#include <bitset>

namespace nntile
{

//! Large enough signed integer for indexing purposes
using Index = int64_t;

// Supported computational types
//! Double precision alias
using fp64_t = double;
//! Single precision alias
using fp32_t = float;
//! Half precision (forward declaration without definition)
using fp16_t = std::bitset<16>;

// Add more types like fp16_t, bf16_t and tf32_t in the future

} // namespace nntile

