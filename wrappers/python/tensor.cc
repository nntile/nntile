/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file wrappers/python/tensor.cc
 * Python module nntile.tensor
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-01-19
 * */

#include "tensor/traits.hh"
#include "tensor/distributions.hh"
#include "tensor/tensor.hh"
#include <pybind11/pybind11.h>
#include "nntile/base_types.hh"

using namespace nntile;
namespace py = pybind11;

// Define nntile.tensor module
PYBIND11_MODULE(tensor, m)
{
    // Wrapper for TensorTraits class
    def_traits(m);
    // Wrapper for submodule distributions
    def_distributions(m);
    // Wrapper for Tensor<T> class
    def_tensor<fp32_t>(m, "Tensor_fp32");
    def_tensor<fp64_t>(m, "Tensor_fp64");
}

