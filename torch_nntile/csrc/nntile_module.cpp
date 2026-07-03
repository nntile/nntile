/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_module.cpp
 */

#include <torch/extension.h>

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "nntile_context.h"
#include "nntile_cross_entropy.h"
#include "nntile_rms_norm.h"
#include "nntile_norm.h"
#include "nntile_graph_recorder.h"
#include "nntile_sgd_step.h"
#include "nntile_adam_step.h"

#ifdef TORCH_NNTILE_USE_LIBNNTILE
#include <nntile/base_types.hh>
#endif

namespace torch_nntile
{

c10::Device nntile_device()
{
    return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

bool is_registered()
{
    return true;
}

bool has_libnntile()
{
#ifdef TORCH_NNTILE_USE_LIBNNTILE
    return true;
#else
    return false;
#endif
}

int64_t buffer_nbytes(const at::Tensor &tensor)
{
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        "buffer_nbytes expects an nntile tensor");
    return static_cast<int64_t>(tensor.storage().nbytes());
}

bool buffer_equal_cpu(const at::Tensor &nntile_tensor, const at::Tensor &cpu_tensor)
{
    TORCH_CHECK(
        nntile_tensor.device().type() == c10::DeviceType::PrivateUse1,
        "buffer_equal_cpu expects nntile tensor as first argument");
    TORCH_CHECK(cpu_tensor.is_cpu(), "buffer_equal_cpu expects CPU tensor");
    // Host read: graph mode requires execute() before nntile -> CPU copy.
    at::Tensor lhs = nntile_tensor.contiguous().cpu();
    at::Tensor rhs = cpu_tensor.contiguous();
    return lhs.equal(rhs);
}

RuntimeMode parse_runtime_mode(const std::string &mode)
{
    std::string lowered = mode;
    std::transform(
        lowered.begin(),
        lowered.end(),
        lowered.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lowered == "eager")
    {
        return RuntimeMode::Eager;
    }
    if (lowered == "graph")
    {
        return RuntimeMode::Graph;
    }
    throw std::runtime_error(
        "torch_nntile.init_context runtime_mode must be 'eager' or 'graph'");
}

void init_context_py(
    int ncpu,
    int ncuda,
    int ooc_enabled,
    const std::string &ooc_path,
    std::size_t ooc_size,
    int logger,
    int verbose,
    bool cpu_fallback,
    const std::string &runtime_mode)
{
    init_context(
        ncpu,
        ncuda,
        ooc_enabled,
        ooc_path.c_str(),
        ooc_size,
        logger,
        verbose,
        cpu_fallback,
        parse_runtime_mode(runtime_mode));
}

std::vector<std::int64_t> parse_tile_sizes_py(const py::object &tile_sizes)
{
    if (py::isinstance<py::int_>(tile_sizes))
    {
        const std::int64_t value = tile_sizes.cast<std::int64_t>();
        if (value <= 0)
        {
            throw std::runtime_error(
                "torch_nntile.set_axis_group_tiling: tile size must be positive");
        }
        return {value};
    }
    if (py::isinstance<py::list>(tile_sizes) ||
        py::isinstance<py::tuple>(tile_sizes))
    {
        std::vector<std::int64_t> sizes;
        for (const py::handle item : tile_sizes)
        {
            const std::int64_t value = py::cast<std::int64_t>(item);
            if (value <= 0)
            {
                throw std::runtime_error(
                    "torch_nntile.set_axis_group_tiling: tile size must be "
                    "positive");
            }
            sizes.push_back(value);
        }
        if (sizes.empty())
        {
            throw std::runtime_error(
                "torch_nntile.set_axis_group_tiling: tile_sizes must be "
                "non-empty");
        }
        return sizes;
    }
    throw std::runtime_error(
        "torch_nntile.set_axis_group_tiling: tile_sizes must be int or "
        "sequence of ints");
}

void set_axis_group_name_py(
    const at::Tensor &tensor,
    const py::dict &names)
{
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        "set_axis_group_name expects an nntile tensor");
    std::unordered_map<int, std::string> parsed;
    for (const auto &item : names)
    {
        const int dim = py::cast<int>(item.first);
        const std::string name = py::cast<std::string>(item.second);
        parsed.emplace(dim, name);
    }
    set_axis_group_name(
        tensor.storage().data_ptr().get(),
        static_cast<int>(tensor.dim()),
        parsed);
}

void set_axis_group_tiling_py(
    const std::string &name,
    const py::object &tile_sizes)
{
    set_axis_group_tiling(name, parse_tile_sizes_py(tile_sizes));
}

} // namespace torch_nntile

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("nntile_device", &torch_nntile::nntile_device, "Return nntile device");
    m.def("is_registered", &torch_nntile::is_registered, "Backend loaded");
    m.def(
        "has_libnntile",
        &torch_nntile::has_libnntile,
        "Whether libnntile TensorGraph add is linked");
    m.def("buffer_nbytes", &torch_nntile::buffer_nbytes, "Storage nbytes");
    m.def(
        "buffer_equal_cpu",
        &torch_nntile::buffer_equal_cpu,
        "Compare nntile tensor to CPU tensor");
    m.def(
        "init_context",
        &torch_nntile::init_context_py,
        "Configure StarPU workers before the first nntile op",
        py::arg("ncpu") = -1,
        py::arg("ncuda") = -1,
        py::arg("ooc_enabled") = 0,
        py::arg("ooc_path") = "/tmp/nntile_ooc",
        py::arg("ooc_size") = 16 * 1024 * 1024,
        py::arg("logger") = 0,
        py::arg("verbose") = 0,
        py::arg("cpu_fallback") = true,
        py::arg("runtime_mode") = "eager");
    m.def(
        "is_cpu_fallback_enabled",
        &torch_nntile::is_cpu_fallback_enabled,
        "Whether unsupported ops may fall back to CPU");
    m.def(
        "is_context_initialized",
        &torch_nntile::is_context_initialized,
        "Whether the libnntile context has been created");
    m.def(
        "restrict_cpu",
        &torch_nntile::restrict_cpu,
        "Run StarPU codelets on CPU workers only");
    m.def(
        "restrict_cuda",
        &torch_nntile::restrict_cuda,
        "Run StarPU codelets on CUDA workers only");
    m.def(
        "restore_where",
        &torch_nntile::restore_where,
        "Restore default StarPU codelet worker placement");
    m.def(
        "wait_for_all",
        &torch_nntile::wait_for_all,
        "Block until all submitted StarPU tasks finish");
    m.def(
        "shutdown_context",
        &torch_nntile::shutdown_context,
        "Shut down libnntile / StarPU (safe to call repeatedly)");
    m.def(
        "execute",
        &torch_nntile::execute_pending_graph,
        "Compile, run, and reset the pending TensorGraph (legacy graph mode)");
    m.def(
        "compile_graph",
        &torch_nntile::compile_graph,
        "Lower and compile the pending TensorGraph into a persistent session");
    m.def(
        "run",
        &torch_nntile::run_graph,
        "Execute the compiled graph session (no data transfer)");
    m.def(
        "reset_graph_session",
        &torch_nntile::reset_graph_session,
        "Discard the compiled graph session and recorder state");
    m.def(
        "has_graph_session",
        &torch_nntile::has_graph_session,
        "Whether a compiled graph session exists");
    m.def(
        "has_pending_graph",
        &torch_nntile::has_pending_graph,
        "Whether a deferred TensorGraph is waiting for execute()");
    m.def(
        "is_graph_mode",
        &torch_nntile::is_graph_mode,
        "Whether runtime_mode is graph");
    m.def(
        "set_axis_group_name",
        &torch_nntile::set_axis_group_name_py,
        "Name TensorGraph axis groups for selected tensor dimensions",
        py::arg("tensor"),
        py::arg("names"));
    m.def(
        "set_axis_group_tiling",
        &torch_nntile::set_axis_group_tiling_py,
        "Set tiling for a named axis group before execute()",
        py::arg("name"),
        py::arg("tile_sizes"));
    m.def(
        "format_axis_groups",
        &torch_nntile::format_axis_groups,
        "Format pending TensorGraph axis groups (like C++ TensorGraph::to_string)");
    m.def(
        "print_axis_groups",
        &torch_nntile::print_axis_groups,
        "Print pending TensorGraph axis groups to stdout");
    m.def(
        "cross_entropy_forward",
        &torch_nntile::cross_entropy_forward,
        "NNTile cross-entropy forward (logits on nntile)",
        py::arg("logits"),
        py::arg("target"),
        py::arg("reduction") = 1,
        py::arg("ignore_index") = -100);
    m.def(
        "cross_entropy_backward",
        &torch_nntile::cross_entropy_backward,
        "NNTile cross-entropy backward w.r.t. logits",
        py::arg("logits"),
        py::arg("target"),
        py::arg("grad_output"),
        py::arg("reduction") = 1,
        py::arg("ignore_index") = -100);
    m.def(
        "sgd_step",
        &torch_nntile::sgd_step,
        "Fused SGD step on nntile tensors (updates param and velocity in-place)",
        py::arg("param"),
        py::arg("grad"),
        py::arg("velocity"),
        py::arg("num_iter"),
        py::arg("lr"),
        py::arg("momentum") = 0.0,
        py::arg("weight_decay") = 0.0,
        py::arg("dampening") = 0.0,
        py::arg("nesterov") = false);
    m.def(
        "adam_step",
        &torch_nntile::adam_step,
        "Fused Adam step on nntile tensors (updates param and moments in-place)",
        py::arg("param"),
        py::arg("grad"),
        py::arg("first_moment"),
        py::arg("second_moment"),
        py::arg("num_iter"),
        py::arg("lr"),
        py::arg("beta_1") = 0.9,
        py::arg("beta_2") = 0.999,
        py::arg("eps") = 1e-8,
        py::arg("weight_decay") = 0.0);
    m.def(
        "adamw_step",
        &torch_nntile::adamw_step,
        "Fused AdamW step on nntile tensors (updates param and moments in-place)",
        py::arg("param"),
        py::arg("grad"),
        py::arg("first_moment"),
        py::arg("second_moment"),
        py::arg("num_iter"),
        py::arg("lr"),
        py::arg("beta_1") = 0.9,
        py::arg("beta_2") = 0.999,
        py::arg("eps") = 1e-8,
        py::arg("weight_decay") = 0.0);
    m.def(
        "rms_norm_forward",
        &torch_nntile::rms_norm_forward,
        "NNTile RMSNorm forward",
        py::arg("input"),
        py::arg("normalized_shape"),
        py::arg("weight") = py::none(),
        py::arg("eps") = py::none());
    m.def(
        "rms_norm_backward",
        &torch_nntile::rms_norm_backward,
        "NNTile RMSNorm backward",
        py::arg("grad_out"),
        py::arg("input"),
        py::arg("normalized_shape"),
        py::arg("rstd"),
        py::arg("weight") = py::none(),
        py::arg("output_mask"));
    m.def(
        "norm_forward",
        &torch_nntile::norm_forward,
        "NNTile 2-norm forward",
        py::arg("input"),
        py::arg("dim") = py::none(),
        py::arg("keepdim") = false);
    m.def(
        "norm_backward",
        &torch_nntile::norm_backward,
        "NNTile 2-norm backward",
        py::arg("grad_out"),
        py::arg("input"),
        py::arg("norm_values"),
        py::arg("dim") = py::none(),
        py::arg("keepdim") = false);
}
