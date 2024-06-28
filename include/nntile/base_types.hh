/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/base_types.hh
 * Base integer and floating point types.
 *
 * @version 1.0.0
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
// class fp64_t {
//     public:
//         double _;
//         double numpy_dtype;
// };
//! Single precision alias
using fp32_t = float;
// class fp32_t {
//     public:
//         float _;
//         float numpy_dtype;
// };

//! Half precision FP16 mock type
class fp16_t
{
private:
    int16_t _;
};

class fp32_fast_tf32_t
{
    public:
        float _;
        // float numpy_dtype;
};

using scal_t = float;

// Boolean type for mask
using bool_t = bool;


// Add more types like fp16_t, bf16_t and tf32_t in the future

} // namespace nntile
