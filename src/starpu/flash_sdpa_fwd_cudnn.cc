/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/flash_sdpa_fwd_cudnn.cc
 * Flash attention scaled dot-product attention forward pass using cuDNN
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/flash_sdpa_fwd_cudnn.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>
#include <algorithm>

// Third-party libraries
#ifdef NNTILE_USE_CUDA
#include <cudnn.h>
#endif

// Other NNTile headers
#include "nntile/kernel/flash_sdpa_fwd_cudnn.hh"
#include "nntile/context.hh"

namespace nntile::starpu
{

// Graphs are cached per codelet instance and reused between tasks

//! Constructor
template<typename T>
FlashSdpaFwdCudnn<std::tuple<T>>::FlashSdpaFwdCudnn():
    codelet("nntile_flash_sdpa_fwd_cudnn", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Destructor
template<typename T>
FlashSdpaFwdCudnn<std::tuple<T>>::~FlashSdpaFwdCudnn()
{
#ifdef NNTILE_USE_CUDA
    for (auto &cache : worker_caches_) {
        cache.reset();
    }
#endif // NNTILE_USE_CUDA
}

//! Apply flash_sdpa_fwd_cudnn on StarPU buffer on CPU (not supported)
template<typename T>
void FlashSdpaFwdCudnn<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // CPU implementation not supported for cuDNN Flash Attention
    // This operation requires CUDA/cuDNN
    // Do nothing - this codelet should only be called on CUDA devices
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply flash_sdpa_fwd_cudnn on StarPU buffer on CUDA
template<typename T>
void FlashSdpaFwdCudnn<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    args_t *args = reinterpret_cast<args_t *>(cl_args);

    auto *owner = args->owner;
    if (owner == nullptr) {
        std::cerr << "CUDA Worker: Codelet instance is null, skipping task" << std::endl;
        return;
    }

    const int worker_id = starpu_worker_get_id();
    if (worker_id < 0 || worker_id >= STARPU_NMAXWORKERS) {
        std::cerr << "CUDA Worker: Invalid worker id " << worker_id << std::endl;
        return;
    }

    auto &worker_cache = owner->get_or_create_worker_cache(worker_id);

    // Get cuDNN handle and CUDA stream
    cudnnHandle_t handle = cudnn_get_local_handle();
    if (handle == nullptr) {
        std::cerr << "CUDA Worker: cuDNN handle is null, skipping task" << std::endl;
        return;
    }

    cudaStream_t stream = starpu_cuda_get_local_stream();
    cudnnSetStream(handle, stream);

    const uint32_t hash = hash_parameters(args->seq, args->head, args->batch);

    auto prepared_graph = owner->find_cached_graph(
        worker_cache,
        hash,
        args->seq,
        args->head,
        args->batch
    );

    if (!prepared_graph)
    {
        prepared_graph = kernel::flash_sdpa_fwd_cudnn::prepare_graph<T>(
            handle,
            args->seq,
            args->head,
            args->batch
        );

        if (!prepared_graph)
        {
            std::cerr << "CUDA Worker: Failed to prepare cuDNN graph" << std::endl;
            return;
        }

        prepared_graph = owner->store_graph(
            worker_cache,
            hash,
            args->seq,
            args->head,
            args->batch,
            prepared_graph
        );
    }

    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *K = interfaces[0]->get_ptr<T>();           // Key
    const T *Q = interfaces[1]->get_ptr<T>();           // Query
    const T *mask = interfaces[2]->get_ptr<T>();        // Mask (always present)
    fp32_t *logsumexp = interfaces[3]->get_ptr<fp32_t>();         // Log-sum-exp
    const T *V = interfaces[4]->get_ptr<T>();           // Value
    T *A = interfaces[5]->get_ptr<T>();                 // Attention output

    // Execute the prepared graph - mask is already nullptr when not used
    kernel::flash_sdpa_fwd_cudnn::execute_graph<T>(
        handle,
        prepared_graph,
        K,
        Q,
        mask,
        logsumexp,
        V,
        A
    );
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for flash_sdpa_fwd_cudnn tasks that depends only on cl_arg
template<typename T>
uint32_t FlashSdpaFwdCudnn<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = hash_parameters(args->seq, args->head, args->batch);
    return hash;
}

template<typename T>
void FlashSdpaFwdCudnn<std::tuple<T>>::submit(Index seq, Index head, Index batch,
        Handle K, Handle Q, Handle mask, Handle logsumexp, Handle V, Handle A)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
#ifdef NNTILE_USE_CUDA
    args->owner = this;
    args->seq = seq;
    args->head = head;
    args->batch = batch;
#endif // NNTILE_USE_CUDA

    // Submit task - always include mask parameter
    int ret = starpu_task_insert(&codelet,
            STARPU_R, K.get(),              // Key
            STARPU_R, Q.get(),              // Query
            STARPU_R, mask.get(),           // Mask
            STARPU_RW, logsumexp.get(),     // Log-sum-exp
            STARPU_R, V.get(),              // Value
            STARPU_RW, A.get(),             // Attention output
            STARPU_CL_ARGS, args, sizeof(*args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in flash_sdpa_fwd_cudnn task submission");
    }
}

#ifdef NNTILE_USE_CUDA
template<typename T>
typename FlashSdpaFwdCudnn<std::tuple<T>>::WorkerCache &
FlashSdpaFwdCudnn<std::tuple<T>>::get_or_create_worker_cache(int worker_id)
{
    std::call_once(worker_cache_flags_[worker_id], [this, worker_id]() {
        worker_caches_[worker_id] = std::make_unique<WorkerCache>();
    });
    return *worker_caches_[worker_id];
}

template<typename T>
kernel::flash_sdpa_fwd_cudnn::FlashSdpaGraph FlashSdpaFwdCudnn<std::tuple<T>>::find_cached_graph(
    WorkerCache &cache,
    uint32_t hash,
    Index seq,
    Index head,
    Index batch
)
{
    auto it = cache.graphs.find(hash);
    if (it == cache.graphs.end())
    {
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

    if (match == it->second.end())
    {
        return {};
    }

    return match->graph;
}

template<typename T>
kernel::flash_sdpa_fwd_cudnn::FlashSdpaGraph FlashSdpaFwdCudnn<std::tuple<T>>::store_graph(
    WorkerCache &cache,
    uint32_t hash,
    Index seq,
    Index head,
    Index batch,
    kernel::flash_sdpa_fwd_cudnn::FlashSdpaGraph graph
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

    if (match != bucket.end())
    {
        return match->graph;
    }

    bucket.push_back(CacheEntry{graph, seq, head, batch});
    return bucket.back().graph;
}

template<typename T>
uint32_t FlashSdpaFwdCudnn<std::tuple<T>>::hash_parameters(Index seq, Index head, Index batch)
{
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&seq, sizeof(seq), hash);
    hash = starpu_hash_crc32c_be_n(&head, sizeof(head), hash);
    hash = starpu_hash_crc32c_be_n(&batch, sizeof(batch), hash);
    return hash;
}
#endif // NNTILE_USE_CUDA

// Explicit instantiation
template class FlashSdpaFwdCudnn<std::tuple<nntile::bf16_t>>;
template class FlashSdpaFwdCudnn<std::tuple<nntile::fp16_t>>;

//! Pack of flash_sdpa_fwd_cudnn operations for different types
flash_sdpa_fwd_cudnn_pack_t flash_sdpa_fwd_cudnn;

} // namespace nntile::starpu
