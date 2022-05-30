/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file wrappers/python/layer/fully_connected.hh
 * Wrapper for fully connected layer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/layer/fully_connected.hh"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace nntile;
namespace py = pybind11;

template<typename T>
void fclayer_from_array(const FullyConnected<T> &layer,
        const py::array_t<T, py::array::f_style | py::array::forcecast> &array)
{
    const Tensor<T> &weight = layer.get_weight();
    if(weight.ndim != array.ndim())
    {
        throw std::runtime_error("weight.ndim != array.ndim()");
    }
    for(Index i = 0; i < weight.ndim; ++i)
    {
        if(array.shape()[i] != weight.shape[i])
        {
            throw std::runtime_error("array.shape()[i] != weight.shape[i]");
        }
    }
    const Tile<T> tile(weight, const_cast<T *>(array.data()), weight.nelems);
    // Blocking version
    copy_intersection(tile, weight);
}

template<typename T>
void fclayer_to_array(const FullyConnected<T> &layer,
        py::array_t<T, py::array::f_style | py::array::forcecast> &array)
{
    const Tensor<T> &weight = layer.get_weight();
    if(weight.ndim != array.ndim())
    {
        throw std::runtime_error("weight.ndim != array.ndim()");
    }
    for(Index i = 0; i < weight.ndim; ++i)
    {
        if(array.shape()[i] != weight.shape[i])
        {
            throw std::runtime_error("array.shape()[i] != weight.shape[i]");
        }
    }
    const Tile<T> tile(weight, array.mutable_data(), weight.nelems);
    // Blocking version
    copy_intersection(weight, tile);
}

template<typename T>
void extend_module(py::module_ m, const char *name)
{
    py::class_<FullyConnected<T>>(m, name).
        def(py::init<const std::vector<Index> &, const std::vector<Index> &>()).
        def("init", py::overload_cast<unsigned long long, T, T>(
                    &FullyConnected<T>::init)).
        def("unregister", &FullyConnected<T>::unregister).
        def("from_array", fclayer_from_array<T>).
        def("to_array", fclayer_to_array<T>).
        def("forward_async", &FullyConnected<T>::forward_async);
    m.def("fclayer_to_array", fclayer_to_array<T>);
    m.def("fclayer_from_array", fclayer_to_array<T>);
}

PYBIND11_MODULE(layer, m)
{
    extend_module<fp32_t>(m, "FC_fp32");
    extend_module<fp64_t>(m, "FC_fp64");
}

