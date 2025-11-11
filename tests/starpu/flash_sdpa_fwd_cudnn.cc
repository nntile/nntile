/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/flash_sdpa_fwd_cudnn.cc
 * Flash Attention SDPA forward operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/starpu/flash_sdpa_fwd_cudnn.hh"
#include "nntile/kernel/flash_sdpa_fwd_cudnn.hh"
#include "../testing.hh"
#ifdef NNTILE_USE_CUDA
#   include <cuda_runtime.h>
#   include <cudnn_frontend.h>
#endif // NNTILE_USE_CUDA
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <limits>
#include <type_traits>

using namespace nntile;
using namespace nntile::starpu;

#ifdef NNTILE_USE_CUDA
template<typename T>
void validate_cuda(Index seq, Index head, Index batch)
{
    using Y = typename T::repr_t;

    // Get a StarPU CUDA worker (to perform computations on the same device)
    int cuda_worker_id = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
    // Choose worker CUDA device
    int dev_id = starpu_worker_get_devid(cuda_worker_id);
    cudaError_t cuda_err = cudaSetDevice(dev_id);
    TEST_ASSERT(cuda_err == cudaSuccess);

    // Create CUDA stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);

    // Create cuDNN handle
    cudnnHandle_t handle;
    cudnnStatus_t cudnn_status = cudnnCreate(&handle);
    TEST_ASSERT(cudnn_status == CUDNN_STATUS_SUCCESS);
    cudnn_status = cudnnSetStream(handle, stream);
    TEST_ASSERT(cudnn_status == CUDNN_STATUS_SUCCESS);

    // Initialize input data (batch here is the combined batch dimension)
    std::vector<T> K(batch * seq * head);
    std::vector<T> Q(batch * seq * head);
    std::vector<T> V(batch * seq * head);
    std::vector<T> mask;

    // Fill with some test values
    for(Index i = 0; i < batch * seq * head; ++i)
    {
        K[i] = T(Y(0.1 * (i % 10 - 5)));
        Q[i] = T(Y(0.1 * ((i + 1) % 10 - 5)));
        V[i] = T(Y(0.1 * ((i + 2) % 10 - 5)));
    }

    // Create custom mask (batch here is the actual batch dimension for mask)
    Index actual_batch = batch / (1 * 1);  // Assuming kv_group_size=1, n_head_kv=1
    mask.resize(actual_batch * seq * seq);
    for(Index b = 0; b < actual_batch; ++b)
    {
        for(Index i = 0; i < seq; ++i)
        {
            for(Index j = 0; j < seq; ++j)
            {
                Index idx = b * seq * seq + i * seq + j;
                // Create a simple mask: allow attention within a window
                if (std::abs(static_cast<long>(i) - static_cast<long>(j)) <= 32)
                {
                    mask[idx] = T(Y(0.0));  // Attend
                }
                else
                {
                    mask[idx] = T(-std::numeric_limits<Y>::infinity());  // Mask
                }
            }
        }
    }

    // Allocate device memory for kernel test
    T *dev_K, *dev_Q, *dev_V, *dev_A_kernel, *dev_mask;
    fp32_t *dev_logsumexp_kernel;
    cuda_err = cudaMalloc(&dev_K, sizeof(T) * batch * seq * head);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_Q, sizeof(T) * batch * seq * head);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_V, sizeof(T) * batch * seq * head);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_A_kernel, sizeof(T) * batch * seq * head);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_logsumexp_kernel, sizeof(fp32_t) * actual_batch * seq);
    TEST_ASSERT(cuda_err == cudaSuccess);

    // Always allocate mask memory (use actual batch size for mask)
    cuda_err = cudaMalloc(&dev_mask, sizeof(T) * actual_batch * seq * seq);
    TEST_ASSERT(cuda_err == cudaSuccess);

    // Copy inputs to device
    cuda_err = cudaMemcpy(dev_K, K.data(), sizeof(T) * batch * seq * head,
                          cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_Q, Q.data(), sizeof(T) * batch * seq * head,
                          cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_V, V.data(), sizeof(T) * batch * seq * head,
                          cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);

    // Always copy mask data (use actual batch size for mask)
    cuda_err = cudaMemcpy(dev_mask, mask.data(), sizeof(T) * actual_batch * seq * seq,
                          cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);

    // Initialize outputs to zero
    cuda_err = cudaMemset(dev_A_kernel, 0, sizeof(T) * batch * seq * head);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemset(dev_logsumexp_kernel, 0, sizeof(fp32_t) * actual_batch * seq);
    TEST_ASSERT(cuda_err == cudaSuccess);

    // Prepare kernel graph once
    auto kernel_graph = kernel::flash_sdpa_fwd_cudnn::prepare_graph<T>(
        handle,
        seq,
        head,
        batch
    );
    TEST_ASSERT(kernel_graph != nullptr);

    // Run kernel directly through the prepared graph
    kernel::flash_sdpa_fwd_cudnn::execute_graph<T>(
        handle,
        kernel_graph,
        dev_K,
        dev_Q,
        dev_mask,
        dev_logsumexp_kernel,
        dev_V,
        dev_A_kernel
    );

    // Synchronize
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);

    // Copy kernel results back to host
    std::vector<T> A_kernel(batch * seq * head);
    std::vector<fp32_t> logsumexp_kernel(actual_batch * seq);
    cuda_err = cudaMemcpy(A_kernel.data(), dev_A_kernel,
                          sizeof(T) * batch * seq * head,
                          cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(logsumexp_kernel.data(), dev_logsumexp_kernel,
                          sizeof(fp32_t) * actual_batch * seq,
                          cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);

    // Destroy prepared graph (release resources)
    kernel_graph.reset();

    // Now test via StarPU submit
    std::vector<T> K2(K), Q2(Q), V2(V), mask2(mask);
    std::vector<T> A_starpu(batch * seq * head, T(Y(0.0)));
    std::vector<fp32_t> logsumexp_starpu(actual_batch * seq, fp32_t(0.0f));

    // Create StarPU handles
    VariableHandle K_handle(&K2[0], sizeof(T) * batch * seq * head);
    VariableHandle Q_handle(&Q2[0], sizeof(T) * batch * seq * head);
    VariableHandle V_handle(&V2[0], sizeof(T) * batch * seq * head);
    VariableHandle A_handle(&A_starpu[0], sizeof(T) * batch * seq * head);
    VariableHandle logsumexp_handle(&logsumexp_starpu[0], sizeof(fp32_t) * actual_batch * seq);

    // Restrict to CUDA and submit
    flash_sdpa_fwd_cudnn.restrict_where(STARPU_CUDA);

    // Always use custom mask (use actual batch size for mask)
    VariableHandle mask_handle(&mask2[0], sizeof(T) * actual_batch * seq * seq);
    flash_sdpa_fwd_cudnn.submit<std::tuple<T>>(
        seq, head, batch,
        K_handle, Q_handle,
        mask_handle,
        logsumexp_handle, V_handle, A_handle
    );
    // Wait for completion
    starpu_task_wait_for_all();
    mask_handle.unregister();

    // Unregister handles
    K_handle.unregister();
    Q_handle.unregister();
    V_handle.unregister();
    A_handle.unregister();
    logsumexp_handle.unregister();

    // Compare results
    Y eps = (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) ? Y(1e-2) : Y(1e-5);
    for(Index i = 0; i < batch * seq * head; ++i)
    {
        Y a_kernel = Y(A_kernel[i]);
        Y a_starpu = Y(A_starpu[i]);
        Y diff = std::abs(a_kernel - a_starpu);
        Y max_val = std::max(std::abs(a_kernel), std::abs(a_starpu));
        if (diff > eps * (Y(1.0) + max_val))
        {
            std::cerr << "Mismatch at index " << i << ": kernel=" << a_kernel
                      << " starpu=" << a_starpu << " diff=" << diff << "\n";
            TEST_ASSERT(false);
        }
    }

    std::cout << "✓ StarPU matches kernel result\n";

    // Cleanup
    cudaFree(dev_K);
    cudaFree(dev_Q);
    cudaFree(dev_V);
    cudaFree(dev_A_kernel);
    cudaFree(dev_logsumexp_kernel);
    cudaFree(dev_mask);
    cudnnDestroy(handle);
    cudaStreamDestroy(stream);
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
    validate_cuda<fp16_t>(64, 32, 1);   // Custom mask
    std::cout << "\n=== Direct kernel call completed successfully! ===\n";

    validate_cuda<bf16_t>(64, 32, 1);   // BF16 with custom mask

    // Medium tests
    std::cout << "\n=== Medium configuration tests ===\n";
    validate_cuda<fp16_t>(256, 64, 2);  // Larger seq, multiple batches
    validate_cuda<bf16_t>(256, 64, 2);   // BF16 with custom mask

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
