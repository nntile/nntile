/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_tensor_gc.cpp
 */

#include "nntile_tensor_gc.h"

#include "nntile_allocator.h"
#include "nntile_graph_recorder_impl.h"

#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/util/Exception.h>

#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace torch_nntile
{

namespace
{

std::mutex g_tensor_gc_mutex;
std::unordered_set<TensorImplKey> g_metadata_only_impls;
std::unordered_set<TensorImplKey> g_staged_input_impls;
std::unordered_set<TensorImplKey> g_tile_persistent_impls;
std::unordered_map<void *, TensorImplKey> g_host_ptr_to_impl;
std::unordered_map<void *, TensorImplKey> g_storage_ctx_to_impl;

} // namespace

TensorImplKey tensor_impl_key(const at::Tensor &tensor)
{
    return tensor.unsafeGetTensorImpl();
}

bool is_metadata_only_tensor(const at::Tensor &tensor)
{
    std::lock_guard<std::mutex> lock(g_tensor_gc_mutex);
    return g_metadata_only_impls.count(tensor_impl_key(tensor)) != 0;
}

bool is_staged_input_tensor(const at::Tensor &tensor)
{
    std::lock_guard<std::mutex> lock(g_tensor_gc_mutex);
    return g_staged_input_impls.count(tensor_impl_key(tensor)) != 0;
}

bool is_staged_input_impl(TensorImplKey impl_key)
{
    std::lock_guard<std::mutex> lock(g_tensor_gc_mutex);
    return g_staged_input_impls.count(impl_key) != 0;
}

bool is_tile_persistent_impl(TensorImplKey impl_key)
{
    std::lock_guard<std::mutex> lock(g_tensor_gc_mutex);
    return g_tile_persistent_impls.count(impl_key) != 0;
}

bool has_host_staging(const at::Tensor &tensor)
{
    if (is_metadata_only_tensor(tensor))
    {
        return false;
    }
    return tensor.storage().nbytes() > 0;
}

void mark_tile_persistent_tensor(const at::Tensor &tensor)
{
    std::lock_guard<std::mutex> lock(g_tensor_gc_mutex);
    const TensorImplKey key = tensor_impl_key(tensor);
    g_tile_persistent_impls.insert(key);
}

void mark_metadata_only_tensor(const at::Tensor &tensor)
{
    std::lock_guard<std::mutex> lock(g_tensor_gc_mutex);
    const TensorImplKey key = tensor_impl_key(tensor);
    g_metadata_only_impls.insert(key);
    g_staged_input_impls.erase(key);
    void *storage_ctx = tensor.storage().data_ptr().get_context();
    if (storage_ctx != nullptr)
    {
        g_storage_ctx_to_impl[storage_ctx] = key;
    }
}

void mark_staged_input_tensor(const at::Tensor &tensor)
{
    std::lock_guard<std::mutex> lock(g_tensor_gc_mutex);
    const TensorImplKey key = tensor_impl_key(tensor);
    g_metadata_only_impls.erase(key);
    g_staged_input_impls.insert(key);
    void *host_ptr = tensor.storage().data_ptr().get();
    if (host_ptr != nullptr)
    {
        g_host_ptr_to_impl[host_ptr] = key;
    }
    void *storage_ctx = tensor.storage().data_ptr().get_context();
    if (storage_ctx != nullptr)
    {
        g_storage_ctx_to_impl.erase(storage_ctx);
    }
}

void clear_tensor_gc_state()
{
    std::lock_guard<std::mutex> lock(g_tensor_gc_mutex);
    g_metadata_only_impls.clear();
    g_staged_input_impls.clear();
    g_tile_persistent_impls.clear();
    g_host_ptr_to_impl.clear();
    g_storage_ctx_to_impl.clear();
}

void on_host_storage_released(void *host_data_ptr, void *storage_ctx)
{
    TensorImplKey released_impl = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_tensor_gc_mutex);
        if (host_data_ptr != nullptr)
        {
            const auto found = g_host_ptr_to_impl.find(host_data_ptr);
            if (found != g_host_ptr_to_impl.end())
            {
                released_impl = found->second;
                g_host_ptr_to_impl.erase(found);
            }
        }
        else if (storage_ctx != nullptr)
        {
            const auto found = g_storage_ctx_to_impl.find(storage_ctx);
            if (found != g_storage_ctx_to_impl.end())
            {
                released_impl = found->second;
                g_storage_ctx_to_impl.erase(found);
            }
        }
        if (released_impl != nullptr)
        {
            g_metadata_only_impls.erase(released_impl);
            g_staged_input_impls.erase(released_impl);
            g_tile_persistent_impls.erase(released_impl);
        }
    }
}

void ensure_host_staging(at::Tensor &tensor)
{
    if (has_host_staging(tensor))
    {
        mark_staged_input_tensor(tensor);
        return;
    }
    const int64_t nbytes = tensor.numel() * tensor.element_size();
    TORCH_CHECK(nbytes >= 0, "ensure_host_staging: invalid tensor size");
    c10::Allocator *allocator = get_nntile_allocator();
    c10::DataPtr data_ptr = allocator->allocate(static_cast<std::size_t>(nbytes));
    auto storage = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        static_cast<std::size_t>(nbytes),
        std::move(data_ptr),
        allocator,
        /*resizable=*/true);
    tensor.unsafeGetTensorImpl()->set_storage_keep_dtype(std::move(storage));
    mark_staged_input_tensor(tensor);
}

} // namespace torch_nntile
