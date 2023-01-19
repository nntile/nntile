/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file wrappers/python/tensor/distributions.cc
 * Wrapper for nntile:tensor::distributions
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-01-19
 * */

#include "tensor/distributions.hh"
#include <nntile/tensor/distributions.hh>
#include <pybind11/stl.h>

using namespace nntile;
using namespace nntile::tensor::distributions;
namespace py = pybind11;

//! Extend pybind11 module by distributions
void def_distributions(py::module_ m)
{
    py::module_ distr = m.def_submodule("distributions");
    distr.def("block_cyclic", &block_cyclic);
}

