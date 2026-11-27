/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_guard.cpp
 */

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace torch_nntile
{

struct NntileGuardImpl final : public c10::impl::DeviceGuardImplInterface
{
    static constexpr c10::DeviceType kDeviceType = c10::DeviceType::PrivateUse1;

    c10::DeviceType type() const override
    {
        return kDeviceType;
    }

    c10::Device exchangeDevice(c10::Device device) const override
    {
        TORCH_INTERNAL_ASSERT(device.type() == kDeviceType);
        TORCH_INTERNAL_ASSERT(device.index() < deviceCount());
        return c10::Device(kDeviceType, 0);
    }

    c10::Device getDevice() const override
    {
        return c10::Device(kDeviceType, 0);
    }

    void setDevice(c10::Device device) const override
    {
        TORCH_INTERNAL_ASSERT(device.type() == kDeviceType);
        TORCH_INTERNAL_ASSERT(device.index() < deviceCount());
    }

    void uncheckedSetDevice(c10::Device /*device*/) const noexcept override
    {
    }

    c10::Stream getStream(c10::Device device) const noexcept override
    {
        return c10::Stream(c10::Stream::DEFAULT, device);
    }

    c10::Stream exchangeStream(c10::Stream /*stream*/) const noexcept override
    {
        return c10::Stream(
            c10::Stream::DEFAULT,
            c10::Device(kDeviceType, 0));
    }

    c10::DeviceIndex deviceCount() const noexcept override
    {
        return 1;
    }
};

} // namespace torch_nntile

C10_REGISTER_GUARD_IMPL(PrivateUse1, torch_nntile::NntileGuardImpl);
