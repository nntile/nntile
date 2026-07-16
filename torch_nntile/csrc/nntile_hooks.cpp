/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_hooks.cpp
 * PrivateUse1 hooks + DeviceGuard so LibTorch autograd can backprop on
 * device=nntile without querying a missing CUDA accelerator.
 */

#include "nntile_allocator.h"
#include "nntile_generator.h"

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace torch_nntile
{

namespace
{

struct NntileHooksInterface final : public at::PrivateUse1HooksInterface
{
    bool isBuilt() const override
    {
        return true;
    }

    bool isAvailable() const override
    {
        return true;
    }

    bool hasPrimaryContext(c10::DeviceIndex /*device_index*/) const override
    {
        return true;
    }

    c10::DeviceIndex deviceCount() const override
    {
        return 1;
    }

    void setCurrentDevice(c10::DeviceIndex /*device*/) const override
    {
    }

    c10::DeviceIndex getCurrentDevice() const override
    {
        return 0;
    }

    c10::DeviceIndex exchangeDevice(c10::DeviceIndex /*device*/) const override
    {
        return 0;
    }

    c10::DeviceIndex maybeExchangeDevice(
        c10::DeviceIndex /*device*/) const override
    {
        return 0;
    }

    const at::Generator &getDefaultGenerator(
        c10::DeviceIndex device_index) const override
    {
        return get_default_nntile_generator(device_index);
    }

    at::Generator getNewGenerator(
        c10::DeviceIndex device_index) const override
    {
        return make_nntile_generator(device_index);
    }

    at::Device getDeviceFromPtr(void * /*data*/) const override
    {
        return c10::Device(c10::DeviceType::PrivateUse1, 0);
    }

    c10::Allocator *getPinnedMemoryAllocator() const override
    {
        return c10::GetCPUAllocator();
    }

    void resizePrivateUse1Bytes(
        const c10::Storage &storage,
        size_t newsize) const override
    {
        c10::StorageImpl *impl = storage.unsafeGetStorageImpl();
        TORCH_CHECK(
            impl->resizable(),
            "Trying to resize non-resizable nntile storage");
        if (newsize == 0)
        {
            impl->set_data_ptr_noswap(
                c10::DataPtr(nullptr, c10::DeviceType::PrivateUse1));
            impl->set_nbytes(0);
            return;
        }
        const auto old_size = impl->nbytes();
        if (newsize <= old_size)
        {
            impl->set_nbytes(newsize);
            return;
        }
        c10::Allocator *allocator = impl->allocator();
        c10::DataPtr new_data = allocator->allocate(newsize);
        if (old_size > 0 && impl->data_ptr())
        {
            allocator->copy_data(new_data.get(), impl->data(), old_size);
        }
        impl->set_data_ptr_noswap(std::move(new_data));
        impl->set_nbytes(newsize);
    }
};

//! Minimal DeviceGuard for PrivateUse1 (no real async streams).
//!
//! LibTorch autograd calls getCurrentStream when accumulating grads on
//! PrivateUse1. Without a registered backend name and guard impl, that
//! path throws even though the test reference is on CPU — the PrivateUse1
//! leaf still goes through accelerator stream bookkeeping.
struct NntileGuardImpl final : public c10::impl::DeviceGuardImplInterface
{
    static constexpr c10::DeviceType static_type =
        c10::DeviceType::PrivateUse1;

    c10::DeviceType type() const override
    {
        return static_type;
    }

    c10::Device exchangeDevice(c10::Device d) const override
    {
        TORCH_INTERNAL_ASSERT(d.type() == static_type);
        return c10::Device(static_type, 0);
    }

    c10::Device getDevice() const override
    {
        return c10::Device(static_type, 0);
    }

    void setDevice(c10::Device d) const override
    {
        TORCH_INTERNAL_ASSERT(d.type() == static_type);
    }

    void uncheckedSetDevice(c10::Device /*d*/) const noexcept override
    {
    }

    c10::Stream getStream(c10::Device d) const noexcept override
    {
        return c10::Stream(c10::Stream::DEFAULT, d);
    }

    c10::Stream getDefaultStream(c10::Device d) const override
    {
        return c10::Stream(c10::Stream::DEFAULT, d);
    }

    c10::Stream getNewStream(
        c10::Device d,
        int /*priority*/ = 0) const override
    {
        return c10::Stream(c10::Stream::DEFAULT, d);
    }

    c10::Stream getStreamFromGlobalPool(
        c10::Device d,
        bool /*isHighPriority*/ = false) const override
    {
        return c10::Stream(c10::Stream::DEFAULT, d);
    }

    c10::Stream exchangeStream(c10::Stream s) const noexcept override
    {
        return s;
    }

    c10::DeviceIndex deviceCount() const noexcept override
    {
        return 1;
    }

    bool queryStream(const c10::Stream & /*stream*/) const override
    {
        return true;
    }

    void synchronizeStream(const c10::Stream & /*stream*/) const override
    {
    }
};

static bool register_nntile_backend [[maybe_unused]] = []()
{
    // Same role as Python
    // torch.utils.rename_privateuse1_backend("nntile").
    // Without this, getAccelerator(true) ignores PrivateUse1.
    if (!c10::is_privateuse1_backend_registered())
    {
        c10::register_privateuse1_backend("nntile");
    }
    at::RegisterPrivateUse1HooksInterface(new NntileHooksInterface());
    return true;
}();

C10_REGISTER_GUARD_IMPL(PrivateUse1, NntileGuardImpl);

} // namespace

} // namespace torch_nntile
