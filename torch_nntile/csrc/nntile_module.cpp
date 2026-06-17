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
#include <stdexcept>
#include <string>

#include "nntile_context.h"
#include "nntile_cross_entropy.h"
#include "nntile_graph_recorder.h"
#include "nntile_sgd_step.h"

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
    // Host read: .cpu() runs execute() in graph mode before copying off-device.
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
        "execute",
        &torch_nntile::execute_pending_graph,
        "Compile and run the pending TensorGraph (graph mode)");
    m.def(
        "has_pending_graph",
        &torch_nntile::has_pending_graph,
        "Whether a deferred TensorGraph is waiting for execute()");
    m.def(
        "is_graph_mode",
        &torch_nntile::is_graph_mode,
        "Whether runtime_mode is graph");
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
}
