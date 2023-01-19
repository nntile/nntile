/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file wrappers/python/tile.cc
 * Python module nntile.tile
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-01-19
 * */

#include "tile/traits.hh"
#include "tile/tile.hh"
#include <pybind11/pybind11.h>
#include "nntile/base_types.hh"

using namespace nntile;
namespace py = pybind11;

// Define nntile.tile module
PYBIND11_MODULE(tile, m)
{
    // Wrapper for TileTraits class
    def_traits(m);
    // Wrapper for Tile<T> class
    def_tile<fp32_t>(m, "Tile_fp32");
    def_tile<fp64_t>(m, "Tile_fp64");
}

