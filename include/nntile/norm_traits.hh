/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * Helper traits for norm result types.
 */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{

template<typename T>
struct norm_value_type
{
    using type = fp32_t;
};

template<>
struct norm_value_type<fp64_t>
{
    using type = fp64_t;
};

template<typename T>
using norm_value_t = typename norm_value_type<T>::type;

} // namespace nntile

