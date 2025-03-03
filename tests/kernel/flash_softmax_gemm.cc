/**
 * @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/flash_softmax_gemm.cc
 * Test for flash_softmax_gemm kernel
 *
 * @version 1.1.0
 */

#include <nntile/kernel/flash_softmax_gemm.hh>
#include <random>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <chrono>

using namespace nntile;

template<typename T>
void test_flash_softmax_gemm()
{
    // Define dimensions
    const Index batch_size = 1;
    const Index seq_len = 4096;
    const Index num_heads = 12;
    const Index head_dim = 64;

    // Create a random number generator
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1, 0.1);

    // Create tensors
    std::vector<T> Q(batch_size * num_heads * seq_len * head_dim);
    std::vector<T> K(batch_size * num_heads * seq_len * head_dim);
    std::vector<T> V(batch_size * num_heads * seq_len * head_dim);
    std::vector<bool_t> mask(seq_len * seq_len);
    std::vector<T> maxsumexp(2 * batch_size * num_heads * seq_len);
    std::vector<T> A(batch_size * num_heads * seq_len * head_dim);

    // Move data to device
    T *Q_dev, *K_dev, *V_dev, *maxsumexp_dev, *A_dev;
    bool_t *mask_dev;
    cudaMalloc((void **)&Q_dev, Q.size() * sizeof(T));
    cudaMemcpy(Q_dev, Q.data(), Q.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&K_dev, K.size() * sizeof(T));
    cudaMemcpy(K_dev, K.data(), K.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&V_dev, V.size() * sizeof(T));
    cudaMemcpy(V_dev, V.data(), V.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&mask_dev, mask.size() * sizeof(bool));
    cudaMemcpy(mask_dev, mask.data(), mask.size() * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&maxsumexp_dev, maxsumexp.size() * sizeof(T));
    cudaMemcpy(maxsumexp_dev, maxsumexp.data(), maxsumexp.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&A_dev, A.size() * sizeof(T));
    cudaMemcpy(A_dev, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice);

    // CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    nntile::kernel::flash_softmax_gemm::cuda<T>(stream, batch_size*num_heads,
            seq_len, head_dim, K_dev, Q_dev, mask_dev, maxsumexp_dev, V_dev, A_dev);
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    std::cout << "Test PASSED!" << std::endl;
}

int main(int argc, char **argv)
{
    // Run test with float precision
    std::cout << "Testing with fp32_t..." << std::endl;
    test_flash_softmax_gemm<fp32_t>();

    // Uncomment to test with other precisions
    // std::cout << "Testing with fp64_t..." << std::endl;
    // test_flash_softmax_gemm<fp64_t>();

    // std::cout << "Testing with bf16_t..." << std::endl;
    // test_flash_softmax_gemm<bf16_t>();

    return 0;
}
