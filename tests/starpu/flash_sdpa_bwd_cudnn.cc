/*! @file tests/starpu/flash_sdpa_bwd_cudnn.cc
 * Flash attention SDPA backward pass on StarPU buffers
 */

#include "nntile/context.hh"
#include "nntile/starpu/flash_sdpa_bwd_cudnn.hh"
#include "nntile/kernel/flash_sdpa_bwd_cudnn.hh"
#include "nntile/kernel/cuda.hh"
#include "../testing.hh"

#ifdef NNTILE_USE_CUDA
#    include <cuda_runtime.h>
#    include <cudnn_frontend.h>
#endif

#include <vector>
#include <limits>
#include <iostream>
#include <cmath>
#include <cstdint>

using namespace nntile;
using namespace nntile::starpu;

#ifdef NNTILE_USE_CUDA

#define CUDNN_CHECK(error, message) \
    do { \
        cudnnStatus_t _err = (error); \
        if (_err != CUDNN_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string(message) + ": cuDNN error code " + std::to_string(_err)); \
        } \
    } while (0)

template<typename T>
void validate_cuda(Index seq, Index head, Index batch)
{
    using Y = typename T::repr_t;

    // Prepare host data
    const Index total = batch * seq * head;
    std::vector<T> K(total);
    std::vector<T> Q(total);
    std::vector<T> V(total);
    std::vector<T> O(total);
    std::vector<T> dO(total);
    std::vector<T> mask(seq * seq);
    std::vector<fp32_t> lse(seq * batch);

    for(Index i = 0; i < total; ++i)
    {
        K[i] = T(Y(0.05 * ((i % 11) - 5)));
        Q[i] = T(Y(0.04 * ((i % 7) - 3)));
        V[i] = T(Y(0.03 * ((i % 13) - 6)));
        O[i] = T(Y(0.03 * ((i % 19) - 9)));
        dO[i] = T(Y(0.02 * (((i + 5) % 9) - 4)));
    }

    for(Index i = 0; i < seq; ++i)
    {
        for(Index j = 0; j < seq; ++j)
        {
            const Index idx = i * seq + j;
            if(std::abs(static_cast<long>(i) - static_cast<long>(j)) <= 4)
            {
                mask[idx] = T(Y(0));
            }
            else
            {
                mask[idx] = T(-std::numeric_limits<Y>::infinity());
            }
        }
    }

    for(Index i = 0; i < seq * batch; ++i)
    {
        lse[i] = ((i % 27) - 13) * 0.1;
    }

    // CUDA setup
    int cuda_worker_id = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
    int dev_id = starpu_worker_get_devid(cuda_worker_id);
    cudaSetDevice(dev_id);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle), "cudnnCreate");
    CUDNN_CHECK(cudnnSetStream(handle, stream), "cudnnSetStream");

    T *dev_K = nullptr, *dev_Q = nullptr, *dev_V = nullptr;
    T *dev_O = nullptr, *dev_dO = nullptr, *dev_mask = nullptr;
    T *dev_dK = nullptr, *dev_dQ = nullptr, *dev_dV = nullptr;
    fp32_t *dev_lse = nullptr;
    void *workspace = nullptr;
    ::int64_t workspace_size = 0;

    CUDA_CHECK(cudaMalloc(&dev_K, sizeof(T) * total), "malloc K");
    CUDA_CHECK(cudaMalloc(&dev_Q, sizeof(T) * total), "malloc Q");
    CUDA_CHECK(cudaMalloc(&dev_V, sizeof(T) * total), "malloc V");
    CUDA_CHECK(cudaMalloc(&dev_O, sizeof(T) * total), "malloc O");
    CUDA_CHECK(cudaMalloc(&dev_dO, sizeof(T) * total), "malloc dO");
    CUDA_CHECK(cudaMalloc(&dev_mask, sizeof(T) * seq * seq), "malloc mask");
    CUDA_CHECK(cudaMalloc(&dev_dK, sizeof(T) * total), "malloc dK");
    CUDA_CHECK(cudaMalloc(&dev_dQ, sizeof(T) * total), "malloc dQ");
    CUDA_CHECK(cudaMalloc(&dev_dV, sizeof(T) * total), "malloc dV");
    CUDA_CHECK(cudaMalloc(&dev_lse, sizeof(fp32_t) * seq * batch), "malloc lse");

    CUDA_CHECK(cudaMemcpy(dev_K, K.data(), sizeof(T) * total, cudaMemcpyHostToDevice), "copy K");
    CUDA_CHECK(cudaMemcpy(dev_Q, Q.data(), sizeof(T) * total, cudaMemcpyHostToDevice), "copy Q");
    CUDA_CHECK(cudaMemcpy(dev_V, V.data(), sizeof(T) * total, cudaMemcpyHostToDevice), "copy V");
    CUDA_CHECK(cudaMemcpy(dev_O, O.data(), sizeof(T) * total, cudaMemcpyHostToDevice), "copy O");
    CUDA_CHECK(cudaMemcpy(dev_dO, dO.data(), sizeof(T) * total, cudaMemcpyHostToDevice), "copy dO");
    CUDA_CHECK(cudaMemcpy(dev_mask, mask.data(), sizeof(T) * seq * seq, cudaMemcpyHostToDevice), "copy mask");
    CUDA_CHECK(cudaMemcpy(dev_lse, lse.data(), sizeof(fp32_t) * seq * batch, cudaMemcpyHostToDevice), "copy lse");
    CUDA_CHECK(cudaMemset(dev_dK, 0, sizeof(T) * total), "memset dK");
    CUDA_CHECK(cudaMemset(dev_dQ, 0, sizeof(T) * total), "memset dQ");
    CUDA_CHECK(cudaMemset(dev_dV, 0, sizeof(T) * total), "memset dV");

    auto bwd_graph = kernel::flash_sdpa_bwd_cudnn::prepare_graph<T>(handle, seq, head, batch);
    TEST_ASSERT(bwd_graph != nullptr);

    auto ws_status = bwd_graph->get_workspace_size(workspace_size);
    TEST_ASSERT(ws_status.is_good());
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, static_cast<size_t>(workspace_size)),
                   "malloc backward workspace");
    }

    kernel::flash_sdpa_bwd_cudnn::execute_graph<T>(
        handle, bwd_graph, dev_K, dev_Q, dev_V, dev_O, dev_dO,
        dev_mask, dev_lse, dev_dK, dev_dQ, dev_dV, workspace);
    CUDA_CHECK(cudaStreamSynchronize(stream), "sync bwd");

    if (workspace != nullptr) {
        CUDA_CHECK(cudaFree(workspace), "free backward workspace");
        workspace = nullptr;
    }

    std::vector<T> dK_ref(total), dQ_ref(total), dV_ref(total);
    CUDA_CHECK(cudaMemcpy(dK_ref.data(), dev_dK, sizeof(T) * total, cudaMemcpyDeviceToHost), "copy dK ref");
    CUDA_CHECK(cudaMemcpy(dQ_ref.data(), dev_dQ, sizeof(T) * total, cudaMemcpyDeviceToHost), "copy dQ ref");
    CUDA_CHECK(cudaMemcpy(dV_ref.data(), dev_dV, sizeof(T) * total, cudaMemcpyDeviceToHost), "copy dV ref");

    // StarPU buffers
    std::vector<T> K_starpu = K;
    std::vector<T> Q_starpu = Q;
    std::vector<T> V_starpu = V;
    std::vector<T> O_starpu = O;
    std::vector<T> dO_starpu = dO;
    std::vector<T> mask_starpu = mask;
    std::vector<fp32_t> lse_starpu = lse;
    std::vector<T> dK_starpu(total, T(Y(0)));
    std::vector<T> dQ_starpu(total, T(Y(0)));
    std::vector<T> dV_starpu(total, T(Y(0)));

    VariableHandle K_handle(K_starpu.data(), sizeof(T) * total);
    VariableHandle Q_handle(Q_starpu.data(), sizeof(T) * total);
    VariableHandle V_handle(V_starpu.data(), sizeof(T) * total);
    VariableHandle O_handle(O_starpu.data(), sizeof(T) * total);
    VariableHandle dO_handle(dO_starpu.data(), sizeof(T) * total);
    VariableHandle mask_handle(mask_starpu.data(), sizeof(T) * seq * seq);
    VariableHandle lse_handle(lse_starpu.data(), sizeof(fp32_t) * seq * batch);
    VariableHandle dK_handle(dK_starpu.data(), sizeof(T) * total);
    VariableHandle dQ_handle(dQ_starpu.data(), sizeof(T) * total);
    VariableHandle dV_handle(dV_starpu.data(), sizeof(T) * total);

    flash_sdpa_bwd_cudnn.restrict_where(STARPU_CUDA);
    flash_sdpa_bwd_cudnn.submit<std::tuple<T>>(
        seq, head, batch,
        K_handle, Q_handle, V_handle,
        O_handle, dO_handle,
        mask_handle, lse_handle,
        dK_handle, dQ_handle, dV_handle);
    starpu_task_wait_for_all();

    K_handle.unregister();
    Q_handle.unregister();
    V_handle.unregister();
    O_handle.unregister();
    dO_handle.unregister();
    mask_handle.unregister();
    lse_handle.unregister();
    dK_handle.unregister();
    dQ_handle.unregister();
    dV_handle.unregister();

    auto compare = [&](const std::vector<T> &ref, const std::vector<T> &got)
    {
        Y eps;
        if constexpr(std::is_same_v<T, bf16_t>)
        {
            eps = Y(1e-2);
        }
        else if constexpr(std::is_same_v<T, fp16_t>)
        {
            eps = Y(2e-3);
        }
        else
        {
            eps = Y(1e-4);
        }
        for(Index i = 0; i < total; ++i)
        {
            Y ref_val = Y(ref[i]);
            Y got_val = Y(got[i]);
            Y diff = std::abs(ref_val - got_val);
            Y max_val = std::max(std::abs(ref_val), std::abs(got_val));
            if (diff > eps * (Y(1.0) + max_val))
            {
                std::cerr << "Mismatch at index " << i << ": ref=" << ref_val
                          << " got=" << got_val << " diff=" << diff << "\n";
                TEST_ASSERT(false);
            }
        }
    };
    compare(dK_ref, dK_starpu);
    compare(dQ_ref, dQ_starpu);
    compare(dV_ref, dV_starpu);

    std::cout << "✓ StarPU matches kernel result\n";

    bwd_graph.reset();
    CUDNN_CHECK(cudnnDestroy(handle), "destroy handle");
    CUDA_CHECK(cudaStreamDestroy(stream), "destroy stream");
    CUDA_CHECK(cudaFree(dev_K), "free K");
    CUDA_CHECK(cudaFree(dev_Q), "free Q");
    CUDA_CHECK(cudaFree(dev_V), "free V");
    CUDA_CHECK(cudaFree(dev_O), "free O");
    CUDA_CHECK(cudaFree(dev_dO), "free dO");
    CUDA_CHECK(cudaFree(dev_mask), "free mask");
    CUDA_CHECK(cudaFree(dev_dK), "free dK");
    CUDA_CHECK(cudaFree(dev_dQ), "free dQ");
    CUDA_CHECK(cudaFree(dev_dV), "free dV");
    CUDA_CHECK(cudaFree(dev_lse), "free lse");
}

#endif // NNTILE_USE_CUDA

int main(int argc, char **argv)
{
    // Initialize StarPU (it will automatically shutdown itself on exit)
    int ncpu=1, ncuda=1, ooc=0, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;

    std::cout << "Initializing NNTile context...\n";
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, 0, nullptr, 0, verbose);

#ifdef NNTILE_USE_CUDA
    // Test with different configurations
    // Small tests
    std::cout << "\n=== Small configuration tests ===\n";
    validate_cuda<fp16_t>(64, 32, 1);
    validate_cuda<bf16_t>(64, 32, 1);

    // Medium tests
    std::cout << "\n=== Medium configuration tests ===\n";
    validate_cuda<fp16_t>(256, 64, 2);
    validate_cuda<bf16_t>(256, 64, 2);

    // Test with different head dimensions
    std::cout << "\n=== Different head dimension tests ===\n";
    validate_cuda<fp16_t>(128, 128, 1);

    std::cout << "\n=== All tests passed! ===\n";
#else
    std::cout << "CUDA not available, skipping tests\n";
    return 1;
#endif // NNTILE_USE_CUDA

    return 0;
}
