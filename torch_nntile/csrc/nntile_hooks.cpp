/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_hooks.cpp
 */

#include "nntile_allocator.h"
#include "nntile_generator.h"

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/StorageImpl.h>

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

static bool register_nntile_hooks [[maybe_unused]] = []() {
    at::RegisterPrivateUse1HooksInterface(new NntileHooksInterface());
    return true;
}();

} // namespace

} // namespace torch_nntile
