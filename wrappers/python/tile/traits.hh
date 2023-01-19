/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file wrappers/python/tile/traits.hh
 * Wrapper for TileTraits
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-01-19
 * */

#pragma once

#include <pybind11/pybind11.h>

// Extend pybind11 module by TileTraits class
void def_traits(pybind11::module_ m);

