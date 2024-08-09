/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/cpu.hh
 * CPU-related compute types
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstdint>

namespace nntile::kernel
{

//! Templated class to derive compute type for each of basic NNTile types
template<typename T>
struct CPUComputeType
{
};

//! Compute type for nntile::int64_t type
template<>
struct CPUComputeType<int64_t>
{
    // nntile::int64_t -> std::int64_t from <cstdint>
    using value = std::int64_t;
};

//! Compute type for nntile::bool_t type
template<>
struct CPUComputeType<bool_t>
{
    // nntile::bool_t -> bool from C++ base
    using value = bool;
};

//! Compute type for nntile::fp64_t type
template<>
struct CPUComputeType<fp64_t>
{
    // nntile::fp64_t -> double from C++ base
    using value = double;
};

//! Compute type for nntile::fp32_t type
template<>
struct CPUComputeType<fp32_t>
{
    // nntile::fp32_t -> double from C++ base
    using value = float;
};

//! Compute type for nntile::fp32_fast_tf32_t type
template<>
struct CPUComputeType<fp32_fast_tf32_t>
{
    // No member `value` here is for a reason: this type shall be manually
    // converted into computing types, as memory-bound operations shall be done
    // in `fp32_t`, while compute-bound operations shall use `fp32_t` as data
    // storage type and `tf32_t` as compute type.
};

} // namespace nntile::kernel
