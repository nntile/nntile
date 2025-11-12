/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/flash_sdpa_fwd_cudnn.hh
 * Flash attention scaled dot-product attention forward pass using cuDNN
 *
 * @version 1.1.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// Standard headers
#include <tuple>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <array>
#include <memory>

// NNTile headers
#include <nntile/starpu/codelet.hh>
#include <nntile/starpu/handle.hh>

#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/flash_sdpa_fwd_cudnn.hh>
#endif // NNTILE_USE_CUDA

namespace nntile::starpu
{

//! Generic wrapper class for flash_sdpa_fwd_cudnn operation is not defined
template<typename T>
class FlashSdpaFwdCudnn;

//! Specialization of wrapper class for flash_sdpa_fwd_cudnn operation via std::tuple
template<typename T>
class FlashSdpaFwdCudnn<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    FlashSdpaFwdCudnn();

    //! Destructor
    ~FlashSdpaFwdCudnn();

    //! Structure for operation arguments
    struct args_t
    {
        FlashSdpaFwdCudnn<std::tuple<T>> *owner;
        Index seq;
        Index head;
        Index batch;
    };

    //! Hash for parameters of the current operation
    static uint32_t hash_parameters(
        Index seq,
        Index head,
        Index batch
    );

    //! Footprint function for the current operation
    static uint32_t footprint(struct starpu_task *task);

    //! Wrapper for a generic CPU implementation (not supported)
    static void cpu(void *buffers[], void *cl_args)
        noexcept;

    //! Array of all wrappers for CPU implementations
    static constexpr func_array cpu_funcs = {
        cpu
    };

#ifdef NNTILE_USE_CUDA
    //! Wrapper for a generic CUDA implementation
    static void cuda(void *buffers[], void *cl_args)
        noexcept;

    //! Array of all wrappers for CUDA implementations
    static constexpr func_array cuda_funcs = {
        cuda
    };
#else // NNTILE_USE_CUDA
    //! Array of all wrappers for CUDA implementations
    static constexpr func_array cuda_funcs = {};
#endif // NNTILE_USE_CUDA

    //! Submit flash_sdpa_fwd_cudnn task
    void submit(
        Index seq,
        Index head,
        Index batch,
        Handle K,
        Handle Q,
        Handle mask,
        Handle logsumexp,
        Handle V,
        Handle A
    );

#ifdef NNTILE_USE_CUDA
private:
    struct CacheEntry
    {
        kernel::flash_sdpa_fwd_cudnn::FlashSdpaGraph graph;
        Index seq;
        Index head;
        Index batch;
    };

    struct WorkerCache
    {
        std::unordered_map<uint32_t, std::vector<CacheEntry>> graphs;
    };

    WorkerCache &get_or_create_worker_cache(int worker_id);

    kernel::flash_sdpa_fwd_cudnn::FlashSdpaGraph find_cached_graph(
        WorkerCache &cache,
        uint32_t hash,
        Index seq,
        Index head,
        Index batch
    );

    kernel::flash_sdpa_fwd_cudnn::FlashSdpaGraph store_graph(
        WorkerCache &cache,
        uint32_t hash,
        Index seq,
        Index head,
        Index batch,
        kernel::flash_sdpa_fwd_cudnn::FlashSdpaGraph graph
    );

    std::array<std::once_flag, STARPU_NMAXWORKERS> worker_cache_flags_{};
    std::array<std::unique_ptr<WorkerCache>, STARPU_NMAXWORKERS> worker_caches_{};
#endif // NNTILE_USE_CUDA
};

//! Pack of flash_sdpa_fwd_cudnn operations for different types
/*! Only FP16 and BF16 are supported due to cuDNN limitations */
using flash_sdpa_fwd_cudnn_pack_t = OperationPack<
    FlashSdpaFwdCudnn,
    std::tuple<nntile::bf16_t>,
    std::tuple<nntile::fp16_t>
>;

//! Pack of flash_sdpa_fwd_cudnn operations for different types
extern flash_sdpa_fwd_cudnn_pack_t flash_sdpa_fwd_cudnn;

} // namespace nntile::starpu
