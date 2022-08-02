/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/args/normalize.hh
 * Arguments for normalize codelet
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-02
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{

//! Structure for arguments
template<typename T>
struct normalize_starpu_args
{
    Index m;
    Index n;
    Index k;
    Index l;
    T eps;
};

} // namespace nntile

