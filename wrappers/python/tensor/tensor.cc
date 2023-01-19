/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file wrappers/python/tensor/tensor.cc
 * Wrapper for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-01-19
 * */

#include "tensor/tensor.hh"
#include <nntile/tensor/tensor.hh>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace nntile;
using namespace nntile::tensor;
namespace py = pybind11;

//template<typename T>
//void tensor_from_array(const Tensor<T> &tensor,
//        const py::array_t<T, py::array::f_style | py::array::forcecast> &array,
//        int mpi_owner, starpu_mpi_tag_t last_tag)
//{
//    if(tensor.ndim != array.ndim())
//    {
//        throw std::runtime_error("tensor.ndim != array.ndim()");
//    }
//    for(Index i = 0; i < tensor.ndim; ++i)
//    {
//        if(array.shape()[i] != tensor.shape[i])
//        {
//            throw std::runtime_error("array.shape()[i] != tensor.shape[i]");
//        }
//    }
//    const TensorTraits src_traits(tensor.shape, tensor.shape);
//    const std::vector<int> src_distr(1);
//    const Tensor<T> src_tensor(tensor, distr);
//    const Tile<T> tile(tensor, const_cast<T *>(array.data()), tensor.nelems);
//    // Blocking version
//    copy_intersection(tile, tensor);
//}

//template<typename T>
//void tensor_to_array(const Tensor<T> &tensor,
//        py::array_t<T, py::array::f_style> &array)
//{
//    if(tensor.ndim != array.ndim())
//    {
//        throw std::runtime_error("tensor.ndim != array.ndim()");
//    }
//    for(Index i = 0; i < tensor.ndim; ++i)
//    {
//        if(array.shape()[i] != tensor.shape[i])
//        {
//            throw std::runtime_error("array.shape()[i] != tensor.shape[i]");
//        }
//    }
//    const Tile<T> tile(tensor.shape, array.mutable_data(), tensor.nelems);
//    // Blocking version
//    copy_intersection(tensor, tile);
//}

template<typename T>
void def_tensor(py::module_ m, const char *name)
{
    py::class_<Tensor<T>, TensorTraits>(m, name).
        def(py::init<const TensorTraits &, const std::vector<int> &,
                starpu_mpi_tag_t &>()).
        def_readonly("next_tag", &Tensor<T>::next_tag);
//        def("unregister", &Tensor<T>::unregister).
//        def("from_array", tensor_from_array<T>);
//        def("to_array", tensor_to_array<T>);
//    m.def("tensor_to_array", tensor_to_array<T>);
//    m.def("tensor_from_array", tensor_to_array<T>);
}

// Explicit instantiation
template
void def_tensor<fp32_t>(py::module_ m, const char *name);

template
void def_tensor<fp64_t>(py::module_ m, const char *name);

