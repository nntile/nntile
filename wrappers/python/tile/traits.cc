/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file wrappers/python/tile/traits.cc
 * Wrapper for TileTraits
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-01-19
 * */

#include "tile/traits.hh"
#include <nntile/tile/traits.hh>
#include <pybind11/stl.h>
#include <sstream>

using namespace nntile;
using namespace nntile::tile;
namespace py = pybind11;

//! Extend pybind11 module by TensorTraits class
void def_traits(py::module_ m)
{
    // Define wrapper for the Class
    py::class_<TileTraits>(m, "TileTraits").
        // Constructor
        def(py::init<const std::vector<Index> &>()).
        // __repr__ function for print(object)
        def("__repr__", [](const TileTraits &data){
                std::stringstream stream;
                stream << data;
                return stream.str();}).
        // Number of dimensions
        def_readonly("ndim", &TileTraits::ndim).
        // Shape of a tile
        def_readonly("shape", &TileTraits::shape).
        // Number of elements of a tile
        def_readonly("nelems", &TileTraits::nelems);
}

