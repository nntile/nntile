/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile.hh
 * Header for Tile<T> class with corresponding operations
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

// Get Tile<T> class, all headers are included explicitly
#include <nntile/base_types.hh>
#include <nntile/tile/traits.hh>
#include <nntile/tile/tile.hh>

// Tile operations
#include <nntile/tile/randn.hh>
#include <nntile/tile/copy.hh>
#include <nntile/tile/gemm.hh>
#include <nntile/tile/bias.hh>
#include <nntile/tile/gelu.hh>
#include <nntile/tile/relu.hh>

