/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_module.cpp
 */

#include <torch/extension.h>

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>

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
    m.def("buffer_nbytes", &torch_nntile::buffer_nbytes, "Storage nbytes");
    m.def(
        "buffer_equal_cpu",
        &torch_nntile::buffer_equal_cpu,
        "Compare nntile tensor to CPU tensor");
}
