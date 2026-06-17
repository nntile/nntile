/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_module.cpp
 */

#include <torch/extension.h>

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>

#include "nntile_context.h"
#include "nntile_cross_entropy.h"

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
    at::Tensor lhs = nntile_tensor.contiguous().cpu();
    at::Tensor rhs = cpu_tensor.contiguous();
    return lhs.equal(rhs);
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
        &torch_nntile::init_context,
        "Configure StarPU workers before the first nntile op",
        py::arg("ncpu") = -1,
        py::arg("ncuda") = -1,
        py::arg("ooc_enabled") = 0,
        py::arg("ooc_path") = "/tmp/nntile_ooc",
        py::arg("ooc_size") = 16 * 1024 * 1024,
        py::arg("logger") = 0,
        py::arg("verbose") = 0,
        py::arg("cpu_fallback") = true);
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
}
