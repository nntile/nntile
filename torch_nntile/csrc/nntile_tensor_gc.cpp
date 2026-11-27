/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_tensor_gc.cpp
 */

#include "nntile_tensor_gc.h"

#include "nntile_graph_recorder_impl.h"

#include <ATen/Tensor.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Storage.h>

#include <mutex>
#include <unordered_map>

namespace torch_nntile
{

namespace
{

std::mutex g_tensor_gc_mutex;
std::unordered_map<void *, TensorImplKey> g_storage_ctx_to_impl;

void register_tensor_storage_ctx_locked(const at::Tensor &tensor)
{
    void *storage_ctx = tensor.storage().data_ptr().get_context();
    if (storage_ctx == nullptr)
    {
        return;
    }
    g_storage_ctx_to_impl[storage_ctx] = tensor_impl_key(tensor);
}

} // namespace

TensorImplKey tensor_impl_key(const at::Tensor &tensor)
{
    return tensor.unsafeGetTensorImpl();
}

bool is_metadata_only_tensor(const at::Tensor &tensor)
{
    return tensor.device().type() == c10::DeviceType::PrivateUse1 &&
        tensor.storage().nbytes() == 0;
}

void clear_tensor_gc_state()
{
    std::lock_guard<std::mutex> lock(g_tensor_gc_mutex);
    g_storage_ctx_to_impl.clear();
}

void on_host_storage_released(void *storage_ctx)
{
    TensorImplKey released_impl = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_tensor_gc_mutex);
        if (storage_ctx != nullptr)
        {
            const auto found = g_storage_ctx_to_impl.find(storage_ctx);
            if (found != g_storage_ctx_to_impl.end())
            {
                released_impl = found->second;
                g_storage_ctx_to_impl.erase(found);
            }
        }
    }
    if (released_impl != nullptr)
    {
        on_tensor_impl_released(released_impl);
    }
}

void register_metadata_tensor_storage(const at::Tensor &tensor)
{
    std::lock_guard<std::mutex> lock(g_tensor_gc_mutex);
    register_tensor_storage_ctx_locked(tensor);
}

} // namespace torch_nntile
