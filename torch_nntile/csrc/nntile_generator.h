/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_generator.h
 */

#pragma once

#include <ATen/core/Generator.h>
#include <c10/core/Device.h>

namespace torch_nntile
{

at::Generator make_nntile_generator(c10::DeviceIndex device_index);

const at::Generator &get_default_nntile_generator(c10::DeviceIndex device_index);

} // namespace torch_nntile
