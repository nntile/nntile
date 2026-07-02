/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_allocator.cpp
 */

#include "nntile_allocator.h"

#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace torch_nntile
{

namespace
{

struct VectorStorage
{
    std::vector<std::uint8_t> bytes;
};

std::atomic<std::int64_t> g_storage_release_count{0};

bool trace_storage_enabled()
{
    static const bool enabled = []() {
        const char *env = std::getenv("TORCH_NNTILE_TRACE_STORAGE");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

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
        auto *storage = static_cast<VectorStorage *>(ctx);
        if (trace_storage_enabled())
        {
            std::cerr << "[torch_nntile storage] release data_ptr="
                      << static_cast<void *>(storage->bytes.data())
                      << " nbytes=" << storage->bytes.size() << '\n';
        }
        delete storage;
        g_storage_release_count.fetch_add(1, std::memory_order_relaxed);
    }
};

NntileAllocator g_nntile_allocator;

c10::Allocator *get_nntile_allocator()
{
    return &g_nntile_allocator;
}

std::int64_t storage_release_count()
{
    return g_storage_release_count.load(std::memory_order_relaxed);
}

void reset_storage_release_count()
{
    g_storage_release_count.store(0, std::memory_order_relaxed);
}

} // namespace torch_nntile

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &torch_nntile::g_nntile_allocator);
