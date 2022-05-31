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

#include "nntile/layer.hh"
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

template<typename T>
void mixer_from_arrays(const Mixer<T> &layer,
        const py::array_t<T, py::array::f_style | py::array::forcecast> &array1,
        const py::array_t<T, py::array::f_style | py::array::forcecast> &array2,
        const py::array_t<T, py::array::f_style | py::array::forcecast> &array3,
        const py::array_t<T, py::array::f_style | py::array::forcecast> &array4
        )
{
    const Tensor<T> &mlp1_weight1 = layer.get_mlp1_weight1();
    const Tensor<T> &mlp1_weight2 = layer.get_mlp1_weight2();
    const Tensor<T> &mlp2_weight1 = layer.get_mlp2_weight1();
    const Tensor<T> &mlp2_weight2 = layer.get_mlp2_weight2();
    if(array1.ndim() != 2)
    {
        throw std::runtime_error("array1.ndim() != 2");
    }
    if(array2.ndim() != 2)
    {
        throw std::runtime_error("array2.ndim() != 2");
    }
    if(array3.ndim() != 2)
    {
        throw std::runtime_error("array3.ndim() != 2");
    }
    if(array4.ndim() != 2)
    {
        throw std::runtime_error("array4.ndim() != 2");
    }
    for(Index i = 0; i < 2; ++i)
    {
        if(array1.shape()[i] != mlp1_weight1.shape[i])
        {
            throw std::runtime_error("array1.shape()[i] != "
                    "mlp1_weight1.shape[i]");
        }
        if(array2.shape()[i] != mlp1_weight2.shape[i])
        {
            throw std::runtime_error("array2.shape()[i] != "
                    "mlp1_weight2.shape[i]");
        }
        if(array3.shape()[i] != mlp2_weight1.shape[i])
        {
            throw std::runtime_error("array3.shape()[i] != "
                    "mlp2_weight1.shape[i]");
        }
        if(array4.shape()[i] != mlp2_weight2.shape[i])
        {
            throw std::runtime_error("array4.shape()[i] != "
                    "mlp2_weight2.shape[i]");
        }
    }
    const Tile<T> tile1(mlp1_weight1, const_cast<T *>(array1.data()),
            mlp1_weight1.nelems),
          tile2(mlp1_weight2, const_cast<T *>(array2.data()),
            mlp1_weight2.nelems),
          tile3(mlp2_weight1, const_cast<T *>(array3.data()),
            mlp2_weight1.nelems),
          tile4(mlp2_weight2, const_cast<T *>(array4.data()),
            mlp2_weight2.nelems);
    // Blocking version
    copy_intersection_async(tile1, mlp1_weight1);
    copy_intersection_async(tile2, mlp1_weight2);
    copy_intersection_async(tile3, mlp2_weight1);
    copy_intersection_async(tile4, mlp2_weight2);
    starpu_task_wait_for_all();
}


template<typename T>
void extend_mixer(py::module_ m, const char *name)
{
    py::class_<Mixer<T>>(m, name).
        def(py::init<Index, Index, Index, Index, Index, Index, Index, Index,
                Index, Index, T>()).
        def("from_arrays", mixer_from_arrays<T>).
        def("forward_async", &Mixer<T>::forward_async).
        def("unregister", &Mixer<T>::unregister);
}

PYBIND11_MODULE(layer, m)
{
    extend_module<fp32_t>(m, "FC_fp32");
    extend_module<fp64_t>(m, "FC_fp64");
    extend_mixer<fp32_t>(m, "Mixer_fp32");
    extend_mixer<fp64_t>(m, "Mixer_fp64");
}

