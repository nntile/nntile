/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/flash_sdpa_bwd_cudnn.hh
 * Flash attention scaled dot-product attention backward pass using cuDNN
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/defs.h>

#include <tuple>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <array>
#include <memory>
#include <utility>
#include <cstdint>

#include <nntile/starpu/codelet.hh>
#include <nntile/starpu/handle.hh>

#ifdef NNTILE_USE_CUDA
#include <cuda_runtime.h>
#include <nntile/kernel/flash_sdpa_bwd_cudnn.hh>
#endif // NNTILE_USE_CUDA

namespace nntile::starpu
{

template<typename T>
class FlashSdpaBwdCudnn;

template<typename T>
class FlashSdpaBwdCudnn<std::tuple<T>>
{
public:
    CodeletTyped<T> codelet;

    FlashSdpaBwdCudnn();
    ~FlashSdpaBwdCudnn();

    struct args_t
    {
        FlashSdpaBwdCudnn<std::tuple<T>> *owner;
        Index seq = 0;
        Index head = 0;
        Index batch = 0;
    };

    static uint32_t hash_parameters(Index seq, Index head, Index batch);
    static uint32_t footprint(struct starpu_task *task);

    static constexpr func_array cpu_funcs = {};

#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {
        cuda
    };
#else
    static constexpr func_array cuda_funcs = {};
#endif

    void submit(
        Index seq,
        Index head,
        Index batch,
        Handle K,
        Handle Q,
        Handle V,
        Handle O,
        Handle dO,
        Handle mask,
        Handle logsumexp,
        Handle dK,
        Handle dQ,
        Handle dV,
        Handle scratch_dK,
        Handle scratch_dQ,
        Handle scratch_dV
    );

#ifdef NNTILE_USE_CUDA
private:
    struct CacheEntry
    {
        kernel::flash_sdpa_bwd_cudnn::FlashSdpaGraph graph;
        void *workspace = nullptr;
        std::int64_t workspace_size = 0;
        Index seq = 0;
        Index head = 0;
        Index batch = 0;

        CacheEntry() = default;
        ~CacheEntry()
        {
            release();
        }

        CacheEntry(const CacheEntry &) = delete;
        CacheEntry &operator=(const CacheEntry &) = delete;

        CacheEntry(CacheEntry &&other) noexcept
        {
            move_from(std::move(other));
        }

        CacheEntry &operator=(CacheEntry &&other) noexcept
        {
            if (this != &other) {
                release();
                move_from(std::move(other));
            }
            return *this;
        }

    private:
        void release()
        {
            if (workspace != nullptr) {
                cudaFree(workspace);
                workspace = nullptr;
                workspace_size = 0;
            }
        }

        void move_from(CacheEntry &&other) noexcept
        {
            graph = std::move(other.graph);
            workspace = other.workspace;
            workspace_size = other.workspace_size;
            seq = other.seq;
            head = other.head;
            batch = other.batch;

            other.workspace = nullptr;
            other.workspace_size = 0;
        }
    };

    struct WorkerCache
    {
        std::unordered_map<uint32_t, std::vector<CacheEntry>> graphs;
    };

    WorkerCache &get_or_create_worker_cache(int worker_id);

    CacheEntry *find_cached_graph(
        WorkerCache &cache,
        uint32_t hash,
        Index seq,
        Index head,
        Index batch
    );

    CacheEntry *store_graph(
        WorkerCache &cache,
        uint32_t hash,
        Index seq,
        Index head,
        Index batch,
        kernel::flash_sdpa_bwd_cudnn::FlashSdpaGraph graph
    );

    std::array<std::once_flag, STARPU_NMAXWORKERS> worker_cache_flags_{};
    std::array<std::unique_ptr<WorkerCache>, STARPU_NMAXWORKERS> worker_caches_{};
#endif
};

using flash_sdpa_bwd_cudnn_pack_t = OperationPack<
    FlashSdpaBwdCudnn,
    std::tuple<nntile::bf16_t>,
    std::tuple<nntile::fp16_t>
>;

extern flash_sdpa_bwd_cudnn_pack_t flash_sdpa_bwd_cudnn;

} // namespace nntile::starpu
