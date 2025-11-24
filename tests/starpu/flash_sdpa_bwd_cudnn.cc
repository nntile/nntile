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
    std::vector<T> A(total);
    std::vector<T> dA(total);
    std::vector<T> mask(seq * seq);
    std::vector<fp32_t> lse(seq * batch);
    std::vector<T> base_dK(total);
    std::vector<T> base_dQ(total);
    std::vector<T> base_dV(total);

    for(Index i = 0; i < total; ++i)
    {
        K[i] = T(Y(0.05 * ((i % 11) - 5)));
        Q[i] = T(Y(0.04 * ((i % 7) - 3)));
        V[i] = T(Y(0.03 * ((i % 13) - 6)));
        A[i] = T(Y(0.03 * ((i % 19) - 9)));
        dA[i] = T(Y(0.02 * (((i + 5) % 9) - 4)));
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

    for(Index i = 0; i < total; ++i)
    {
        base_dK[i] = T(Y(0.02 * ((i % 5) - 2)));
        base_dQ[i] = T(Y(0.03 * (((i + 7) % 11) - 5)));
        base_dV[i] = T(Y(0.015 * (((i + 3) % 13) - 6)));
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
    T *dev_A = nullptr, *dev_dA = nullptr, *dev_mask = nullptr;
    T *dev_dK = nullptr, *dev_dQ = nullptr, *dev_dV = nullptr;
    T *dev_scratch_dK = nullptr, *dev_scratch_dQ = nullptr, *dev_scratch_dV = nullptr;
    fp32_t *dev_lse = nullptr;
    void *workspace = nullptr;
    ::int64_t workspace_size = 0;

    CUDA_CHECK(cudaMalloc(&dev_K, sizeof(T) * total), "malloc K");
    CUDA_CHECK(cudaMalloc(&dev_Q, sizeof(T) * total), "malloc Q");
    CUDA_CHECK(cudaMalloc(&dev_V, sizeof(T) * total), "malloc V");
    CUDA_CHECK(cudaMalloc(&dev_A, sizeof(T) * total), "malloc A");
    CUDA_CHECK(cudaMalloc(&dev_dA, sizeof(T) * total), "malloc dA");
    CUDA_CHECK(cudaMalloc(&dev_mask, sizeof(T) * seq * seq), "malloc mask");
    CUDA_CHECK(cudaMalloc(&dev_dK, sizeof(T) * total), "malloc dK");
    CUDA_CHECK(cudaMalloc(&dev_dQ, sizeof(T) * total), "malloc dQ");
    CUDA_CHECK(cudaMalloc(&dev_dV, sizeof(T) * total), "malloc dV");
    CUDA_CHECK(cudaMalloc(&dev_scratch_dK, sizeof(T) * total), "malloc scratch dK");
    CUDA_CHECK(cudaMalloc(&dev_scratch_dQ, sizeof(T) * total), "malloc scratch dQ");
    CUDA_CHECK(cudaMalloc(&dev_scratch_dV, sizeof(T) * total), "malloc scratch dV");
    CUDA_CHECK(cudaMalloc(&dev_lse, sizeof(fp32_t) * seq * batch), "malloc lse");

    CUDA_CHECK(cudaMemcpy(dev_K, K.data(), sizeof(T) * total, cudaMemcpyHostToDevice), "copy K");
    CUDA_CHECK(cudaMemcpy(dev_Q, Q.data(), sizeof(T) * total, cudaMemcpyHostToDevice), "copy Q");
    CUDA_CHECK(cudaMemcpy(dev_V, V.data(), sizeof(T) * total, cudaMemcpyHostToDevice), "copy V");
    CUDA_CHECK(cudaMemcpy(dev_A, A.data(), sizeof(T) * total, cudaMemcpyHostToDevice), "copy A");
    CUDA_CHECK(cudaMemcpy(dev_dA, dA.data(), sizeof(T) * total, cudaMemcpyHostToDevice), "copy dA");
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

    std::vector<T> dK_delta(total), dQ_delta(total), dV_delta(total);
    std::vector<T> dK_ref(total), dQ_ref(total), dV_ref(total);

    CUDA_CHECK(cudaMemset(dev_dK, 0, sizeof(T) * total), "memset dK zero");
    CUDA_CHECK(cudaMemset(dev_dQ, 0, sizeof(T) * total), "memset dQ zero");
    CUDA_CHECK(cudaMemset(dev_dV, 0, sizeof(T) * total), "memset dV zero");

    kernel::flash_sdpa_bwd_cudnn::execute_graph<T>(
        handle, bwd_graph, seq, head, batch,
        dev_K, dev_Q, dev_V, dev_A, dev_dA,
        dev_mask, dev_lse,
        dev_scratch_dK, dev_scratch_dQ, dev_scratch_dV,
        dev_dK, dev_dQ, dev_dV, workspace);
    CUDA_CHECK(cudaStreamSynchronize(stream), "sync delta bwd");

    CUDA_CHECK(cudaMemcpy(dK_delta.data(), dev_dK, sizeof(T) * total, cudaMemcpyDeviceToHost), "copy dK delta");
    CUDA_CHECK(cudaMemcpy(dQ_delta.data(), dev_dQ, sizeof(T) * total, cudaMemcpyDeviceToHost), "copy dQ delta");
    CUDA_CHECK(cudaMemcpy(dV_delta.data(), dev_dV, sizeof(T) * total, cudaMemcpyDeviceToHost), "copy dV delta");

    CUDA_CHECK(cudaMemcpy(dev_dK, base_dK.data(), sizeof(T) * total, cudaMemcpyHostToDevice), "copy base dK");
    CUDA_CHECK(cudaMemcpy(dev_dQ, base_dQ.data(), sizeof(T) * total, cudaMemcpyHostToDevice), "copy base dQ");
    CUDA_CHECK(cudaMemcpy(dev_dV, base_dV.data(), sizeof(T) * total, cudaMemcpyHostToDevice), "copy base dV");

    kernel::flash_sdpa_bwd_cudnn::execute_graph<T>(
        handle, bwd_graph, seq, head, batch,
        dev_K, dev_Q, dev_V, dev_A, dev_dA,
        dev_mask, dev_lse,
        dev_scratch_dK, dev_scratch_dQ, dev_scratch_dV,
        dev_dK, dev_dQ, dev_dV, workspace);
    CUDA_CHECK(cudaStreamSynchronize(stream), "sync bwd");

    if (workspace != nullptr) {
        CUDA_CHECK(cudaFree(workspace), "free backward workspace");
        workspace = nullptr;
    }

    CUDA_CHECK(cudaMemcpy(dK_ref.data(), dev_dK, sizeof(T) * total, cudaMemcpyDeviceToHost), "copy dK ref");
    CUDA_CHECK(cudaMemcpy(dQ_ref.data(), dev_dQ, sizeof(T) * total, cudaMemcpyDeviceToHost), "copy dQ ref");
    CUDA_CHECK(cudaMemcpy(dV_ref.data(), dev_dV, sizeof(T) * total, cudaMemcpyDeviceToHost), "copy dV ref");

    // StarPU buffers
    auto verify_addition = [&](const std::vector<T> &total_vec,
                               const std::vector<T> &delta,
                               const std::vector<T> &base,
                               const char *label)
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
            const Y expected = Y(base[i]) + Y(delta[i]);
            const Y got = Y(total_vec[i]);
            const Y diff = std::abs(expected - got);
            const Y max_val = std::max(std::abs(expected), std::abs(got));
            if (diff > eps * (Y(1.0) + max_val))
            {
                std::cerr << "Addition mismatch (" << label << ") at index " << i
                          << ": expected=" << expected << " got=" << got << "\n";
                TEST_ASSERT(false);
            }
        }
    };

    verify_addition(dK_ref, dK_delta, base_dK, "kernel dK");
    verify_addition(dQ_ref, dQ_delta, base_dQ, "kernel dQ");
    verify_addition(dV_ref, dV_delta, base_dV, "kernel dV");

    std::vector<T> K_starpu = K;
    std::vector<T> Q_starpu = Q;
    std::vector<T> V_starpu = V;
    std::vector<T> A_starpu = A;
    std::vector<T> dA_starpu = dA;
    std::vector<T> mask_starpu = mask;
    std::vector<fp32_t> lse_starpu = lse;
    std::vector<T> dK_starpu = base_dK;
    std::vector<T> dQ_starpu = base_dQ;
    std::vector<T> dV_starpu = base_dV;

    VariableHandle K_handle(K_starpu.data(), sizeof(T) * total);
    VariableHandle Q_handle(Q_starpu.data(), sizeof(T) * total);
    VariableHandle V_handle(V_starpu.data(), sizeof(T) * total);
    VariableHandle A_handle(A_starpu.data(), sizeof(T) * total);
    VariableHandle dA_handle(dA_starpu.data(), sizeof(T) * total);
    VariableHandle mask_handle(mask_starpu.data(), sizeof(T) * seq * seq);
    VariableHandle lse_handle(lse_starpu.data(), sizeof(fp32_t) * seq * batch);
    VariableHandle dK_handle(dK_starpu.data(), sizeof(T) * total);
    VariableHandle dQ_handle(dQ_starpu.data(), sizeof(T) * total);
    VariableHandle dV_handle(dV_starpu.data(), sizeof(T) * total);
    VariableHandle scratch_dK(sizeof(T) * total);
    VariableHandle scratch_dQ(sizeof(T) * total);
    VariableHandle scratch_dV(sizeof(T) * total);

    flash_sdpa_bwd_cudnn.restrict_where(STARPU_CUDA);
    flash_sdpa_bwd_cudnn.submit<std::tuple<T>>(
        seq, head, batch,
        K_handle, Q_handle, V_handle,
        A_handle, dA_handle,
        mask_handle, lse_handle,
        dK_handle, dQ_handle, dV_handle,
        scratch_dK, scratch_dQ, scratch_dV);
    starpu_task_wait_for_all();
    scratch_dK.unregister_submit();
    scratch_dQ.unregister_submit();
    scratch_dV.unregister_submit();

    K_handle.unregister();
    Q_handle.unregister();
    V_handle.unregister();
    A_handle.unregister();
    dA_handle.unregister();
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

    verify_addition(dK_starpu, dK_delta, base_dK, "starpu dK");
    verify_addition(dQ_starpu, dQ_delta, base_dQ, "starpu dQ");
    verify_addition(dV_starpu, dV_delta, base_dV, "starpu dV");

    std::cout << "✓ StarPU matches kernel result\n";

    bwd_graph.reset();
    CUDNN_CHECK(cudnnDestroy(handle), "destroy handle");
    CUDA_CHECK(cudaStreamDestroy(stream), "destroy stream");
    CUDA_CHECK(cudaFree(dev_K), "free K");
    CUDA_CHECK(cudaFree(dev_Q), "free Q");
    CUDA_CHECK(cudaFree(dev_V), "free V");
    CUDA_CHECK(cudaFree(dev_A), "free A");
    CUDA_CHECK(cudaFree(dev_dA), "free dA");
    CUDA_CHECK(cudaFree(dev_mask), "free mask");
    CUDA_CHECK(cudaFree(dev_dK), "free dK");
    CUDA_CHECK(cudaFree(dev_dQ), "free dQ");
    CUDA_CHECK(cudaFree(dev_dV), "free dV");
    CUDA_CHECK(cudaFree(dev_scratch_dK), "free scratch dK");
    CUDA_CHECK(cudaFree(dev_scratch_dQ), "free scratch dQ");
    CUDA_CHECK(cudaFree(dev_scratch_dV), "free scratch dV");
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
