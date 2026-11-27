/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_platform_bind.h
 * Pybind wrappers for platform Ingress / Flush host-copy hooks.
 */

#pragma once

#include "nntile_graph_recorder.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>

namespace py = pybind11;

namespace torch_nntile
{

inline void set_platform_hooks_py(py::object ingress, py::object flush)
{
    PlatformIngressHook in =
        [ingress](
            std::int64_t node_id,
            void const *host,
            std::size_t nbytes,
            std::vector<std::int64_t> const &shape,
            std::string const &dtype)
        {
            py::gil_scoped_acquire gil;
            char const *raw =
                nbytes == 0 ? "" : static_cast<char const *>(host);
            py::bytes payload(raw, static_cast<py::size_t>(nbytes));
            ingress(node_id, payload, shape, dtype);
        };
    PlatformFlushHook out =
        [flush](
            std::string const &phase_json,
            std::vector<std::int64_t> const &gather_ids,
            bool wait_only) -> std::string
        {
            py::gil_scoped_acquire gil;
            py::object result = flush(
                phase_json,
                gather_ids,
                wait_only);
            if (wait_only || result.is_none())
            {
                return {};
            }
            py::bytes payload = py::bytes(result);
            return std::string(payload);
        };
    set_platform_hooks(std::move(in), std::move(out));
}

inline void bind_platform_hooks(py::module_ &m)
{
    m.def(
        "set_platform_hooks",
        &set_platform_hooks_py,
        "Install Ingress / Flush hooks (no local StarPU compile)",
        py::arg("ingress"),
        py::arg("flush"));
    m.def(
        "clear_platform_hooks",
        &clear_platform_hooks,
        "Drop platform Ingress / Flush hooks");
    m.def(
        "platform_session_active",
        &platform_session_active,
        "Whether .to() flushes to a platform client");
}

} // namespace torch_nntile
