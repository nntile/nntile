/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file python/nntile/_bindings/nntile_models.cc
 * Model / optimizer / I/O pybind stubs (extend for full 7-family API).
 *
 * @version 1.1.0
 * */

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_nntile_models(py::module_ &m)
{
    m.doc() = "NNTile model bindings (extend with GPT-2, Llama, BERT, ...)";
}

// Registered from nntile.cc via bind_nntile_models(m).
