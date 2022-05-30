/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file wrappers/python/starpu.hh
 * Wrapper for Starpu
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/starpu.hh"
#include <pybind11/pybind11.h>

using namespace nntile;
namespace py = pybind11;

PYBIND11_MODULE(starpu, m)
{
    py::class_<Starpu>(m, "Starpu").
        def(py::init<>()).
        def_static("wait_for_all", Starpu::wait_for_all);
}
