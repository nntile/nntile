/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/flash_sdpa_bwd_cudnn.cc
 * Flash attention scaled dot-product attention backward pass using cuDNN
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/flash_sdpa_bwd_cudnn.hh"

#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <iostream>

#ifdef NNTILE_USE_CUDA
#include <cudnn.h>
#include <cuda_runtime.h>
#endif

#include "nntile/kernel/flash_sdpa_bwd_cudnn.hh"
#include "nntile/context.hh"

namespace nntile::starpu
{

template<typename T>
FlashSdpaBwdCudnn<std::tuple<T>>::FlashSdpaBwdCudnn():
    codelet("nntile_flash_sdpa_bwd_cudnn", footprint, cpu_funcs, cuda_funcs)
{
}

template<typename T>
FlashSdpaBwdCudnn<std::tuple<T>>::~FlashSdpaBwdCudnn()
{
#ifdef NNTILE_USE_CUDA
    for (auto &cache : worker_caches_) {
        cache.reset();
    }
#endif
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void FlashSdpaBwdCudnn<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID
    auto *args = reinterpret_cast<args_t *>(cl_args);
    if (args == nullptr || args->owner == nullptr) {
        std::cerr << "CUDA Worker: flash_sdpa_bwd_cudnn args are null"
                  << std::endl;
        return;
    }

    const int worker_id = starpu_worker_get_id();
    if (worker_id < 0 || worker_id >= STARPU_NMAXWORKERS) {
        std::cerr << "CUDA Worker: Invalid worker id " << worker_id << std::endl;
        return;
    }

    auto &worker_cache = args->owner->get_or_create_worker_cache(worker_id);

    cudnnHandle_t handle = cudnn_get_local_handle();
    if (handle == nullptr) {
        std::cerr << "CUDA Worker: cuDNN handle is null" << std::endl;
        return;
    }

    cudaStream_t stream = starpu_cuda_get_local_stream();
    cudnnSetStream(handle, stream);

    const uint32_t hash = hash_parameters(args->seq, args->head, args->batch);
    auto *cache_entry = args->owner->find_cached_graph(
        worker_cache, hash, args->seq, args->head, args->batch);

    if (cache_entry == nullptr) {
        auto prepared_graph = kernel::flash_sdpa_bwd_cudnn::prepare_graph<T>(
            handle, args->seq, args->head, args->batch);

        if (!prepared_graph) {
            std::cerr << "CUDA Worker: Failed to prepare cuDNN backward graph"
                      << std::endl;
            return;
        }

        cache_entry = args->owner->store_graph(
            worker_cache, hash, args->seq, args->head, args->batch,
            prepared_graph);
        if (cache_entry == nullptr) {
            std::cerr << "CUDA Worker: Failed to cache cuDNN backward graph" << std::endl;
            return;
        }
    }

    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *K = interfaces[0]->get_ptr<T>();
    const T *Q = interfaces[1]->get_ptr<T>();
    const T *V = interfaces[2]->get_ptr<T>();
    const T *A = interfaces[3]->get_ptr<T>();
    const T *dA = interfaces[4]->get_ptr<T>();
    const T *mask = interfaces[5]->get_ptr<T>();
    const fp32_t *logsumexp = interfaces[6]->get_ptr<fp32_t>();
    T *dK = interfaces[7]->get_ptr<T>();
    T *dQ = interfaces[8]->get_ptr<T>();
    T *dV = interfaces[9]->get_ptr<T>();
    T *scratch_dK = interfaces[10]->get_ptr<T>();
    T *scratch_dQ = interfaces[11]->get_ptr<T>();
    T *scratch_dV = interfaces[12]->get_ptr<T>();

    kernel::flash_sdpa_bwd_cudnn::execute_graph<T>(
        handle,
        cache_entry->graph,
        args->seq,
        args->head,
        args->batch,
        K,
        Q,
        V,
        A,
        dA,
        mask,
        logsumexp,
        scratch_dK,
        scratch_dQ,
        scratch_dV,
        dK,
        dQ,
        dV,
        cache_entry->workspace
    );
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

template<typename T>
uint32_t FlashSdpaBwdCudnn<std::tuple<T>>::hash_parameters(
    Index seq, Index head, Index batch)
{
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&seq, sizeof(seq), hash);
    hash = starpu_hash_crc32c_be_n(&head, sizeof(head), hash);
    hash = starpu_hash_crc32c_be_n(&batch, sizeof(batch), hash);
    return hash;
}

template<typename T>
uint32_t FlashSdpaBwdCudnn<std::tuple<T>>::footprint(struct starpu_task *task)
{
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = hash_parameters(args->seq, args->head, args->batch);
    return hash;
}

template<typename T>
void FlashSdpaBwdCudnn<std::tuple<T>>::submit(Index seq, Index head,
        Index batch, Handle K, Handle Q, Handle V, Handle A, Handle dA,
        Handle mask, Handle logsumexp, Handle dK, Handle dQ, Handle dV,
        Handle scratch_dK, Handle scratch_dQ, Handle scratch_dV)
{
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->owner = this;
    args->seq = seq;
    args->head = head;
    args->batch = batch;

    int ret = starpu_task_insert(&codelet,
            STARPU_R, K.get(),
            STARPU_R, Q.get(),
            STARPU_R, V.get(),
            STARPU_R, A.get(),
            STARPU_R, dA.get(),
            STARPU_R, mask.get(),
            STARPU_R, logsumexp.get(),
            STARPU_RW, dK.get(),
            STARPU_RW, dQ.get(),
            STARPU_RW, dV.get(),
            STARPU_SCRATCH, scratch_dK.get(),
            STARPU_SCRATCH, scratch_dQ.get(),
            STARPU_SCRATCH, scratch_dV.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            0);
    if(ret != 0)
    {
        throw std::runtime_error("Error in flash_sdpa_bwd_cudnn task submission");
    }
}

#ifdef NNTILE_USE_CUDA
template<typename T>
typename FlashSdpaBwdCudnn<std::tuple<T>>::WorkerCache &
FlashSdpaBwdCudnn<std::tuple<T>>::get_or_create_worker_cache(int worker_id)
{
    std::call_once(
        worker_cache_flags_[worker_id],
        [this, worker_id]()
        {
            worker_caches_[worker_id] = std::make_unique<WorkerCache>();
        }
    );
    return *worker_caches_[worker_id];
}

template<typename T>
typename FlashSdpaBwdCudnn<std::tuple<T>>::CacheEntry *
FlashSdpaBwdCudnn<std::tuple<T>>::find_cached_graph(
    WorkerCache &cache,
    uint32_t hash,
    Index seq,
    Index head,
    Index batch
)
{
    auto it = cache.graphs.find(hash);
    if (it == cache.graphs.end()) {
        return {};
    }

    const auto match = std::find_if(
        it->second.begin(),
        it->second.end(),
        [seq, head, batch](const CacheEntry &entry)
        {
            return entry.seq == seq && entry.head == head && entry.batch == batch;
        }
    );

    if (match == it->second.end()) {
        return {};
    }

    return &(*match);
}

template<typename T>
typename FlashSdpaBwdCudnn<std::tuple<T>>::CacheEntry *
FlashSdpaBwdCudnn<std::tuple<T>>::store_graph(
    WorkerCache &cache,
    uint32_t hash,
    Index seq,
    Index head,
    Index batch,
    kernel::flash_sdpa_bwd_cudnn::FlashSdpaBwdGraph graph
)
{
    auto &bucket = cache.graphs[hash];
    const auto match = std::find_if(
        bucket.begin(),
        bucket.end(),
        [seq, head, batch](const CacheEntry &entry)
        {
            return entry.seq == seq && entry.head == head && entry.batch == batch;
        }
    );

    if (match != bucket.end()) {
        return &(*match);
    }

    CacheEntry entry;
    entry.graph = std::move(graph);
    entry.seq = seq;
    entry.head = head;
    entry.batch = batch;

    ::int64_t workspace_size = 0;
    auto status = entry.graph->get_workspace_size(workspace_size);
    if (!status.is_good())
    {
        std::cerr << "CUDA Worker: Failed to query backward workspace size" << std::endl;
        return nullptr;
    }
    entry.workspace_size = workspace_size;
    if (entry.workspace_size > 0) {
        auto cuda_status = cudaMalloc(&entry.workspace,
                                      static_cast<size_t>(entry.workspace_size));
        if (cuda_status != cudaSuccess) {
            std::cerr << "CUDA Worker: Failed to allocate backward workspace" << std::endl;
            return nullptr;
        }
    }

    bucket.push_back(std::move(entry));
    return &bucket.back();
}
#endif // NNTILE_USE_CUDA

template class FlashSdpaBwdCudnn<std::tuple<nntile::bf16_t>>;
template class FlashSdpaBwdCudnn<std::tuple<nntile::fp16_t>>;

flash_sdpa_bwd_cudnn_pack_t flash_sdpa_bwd_cudnn;

} // namespace nntile::starpu
