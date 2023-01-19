/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file wrappers/python/tensor/traits.cc
 * Wrapper for TensorTraits
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-01-19
 * */

#include "tensor/traits.hh"
#include <nntile/tensor/traits.hh>
#include <pybind11/stl.h>
#include <sstream>

using namespace nntile;
using namespace nntile::tensor;
namespace py = pybind11;

//! Extend pybind11 module by TensorTraits class
void def_traits(py::module_ m)
{
    // Define wrapper for the Class
    py::class_<TensorTraits>(m, "TensorTraits").
        // Constructor
        def(py::init<const std::vector<Index> &,
                const std::vector<Index> &>()).
        // __repr__ function for print(object)
        def("__repr__", [](const TensorTraits &data){
                std::stringstream stream;
                stream << data;
                return stream.str();}).
        // Shape of corresponding tile
        def("get_tile_shape", &TensorTraits::get_tile_shape).
        // Shape of a grid
        def("get_grid_shape", [](const TensorTraits &data){
                return data.grid.shape;});
}

