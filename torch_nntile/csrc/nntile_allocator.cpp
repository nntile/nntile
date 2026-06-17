/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_allocator.cpp
 */

#include "nntile_allocator.h"

#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>

#include <cstring>
#include <vector>

namespace torch_nntile
{

namespace
{

struct VectorStorage
{
    std::vector<std::uint8_t> bytes;
};

} // namespace

struct NntileAllocator final : c10::Allocator
{
    c10::DataPtr allocate(std::size_t nbytes) override
    {
        auto *storage = new VectorStorage();
        storage->bytes.resize(nbytes);
        return c10::DataPtr(
            storage->bytes.data(),
            storage,
            &NntileAllocator::release_storage,
            c10::Device(c10::DeviceType::PrivateUse1, 0));
    }

    void copy_data(
        void *dest,
        const void *src,
        std::size_t count) const override
    {
        if (count == 0)
        {
            return;
        }
        std::memcpy(dest, src, count);
    }

    c10::DeleterFnPtr raw_deleter() const override
    {
        return &NntileAllocator::release_storage;
    }

    static void release_storage(void *ctx)
    {
        delete static_cast<VectorStorage *>(ctx);
    }
};

NntileAllocator g_nntile_allocator;

c10::Allocator *get_nntile_allocator()
{
    return &g_nntile_allocator;
}

} // namespace torch_nntile

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &torch_nntile::g_nntile_allocator);
