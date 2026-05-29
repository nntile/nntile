/*! @file python/nntile/_bindings/nntile_models.cc
 * Model / optimizer / I/O pybind stubs (extend for full 7-family API).
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_nntile_models(py::module_ &m)
{
    m.doc() = "NNTile model bindings (extend with GPT-2, Llama, BERT, ...)";
}

// Registered from nntile.cc via bind_nntile_models(m).
