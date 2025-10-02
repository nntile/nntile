/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/adam_step.cc
 * Fused Adam optimizer step
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/adam_step.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <cmath>

using namespace nntile;
using namespace nntile::kernel;
using namespace nntile::kernel::adam_step;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index num_iter, Index num_elems, Scalar beta_1, Scalar beta_2,
    Scalar eps, Scalar lr, Scalar weight_decay, const std::vector<T>& grad,
    std::vector<T>& first_moment, std::vector<T>& second_moment,
    std::vector<T>& p)
{
    // Copy to device
    T *dev_grad, *dev_first_moment, *dev_second_moment, *dev_p;
    cudaError_t cuda_err = cudaMalloc(&dev_grad, sizeof(T)*num_elems);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_first_moment, sizeof(T)*num_elems);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_second_moment, sizeof(T)*num_elems);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_p, sizeof(T)*num_elems);
    TEST_ASSERT(cuda_err == cudaSuccess);

    cuda_err = cudaMemcpy(dev_grad, &grad[0], sizeof(T)*num_elems, cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_first_moment, &first_moment[0], sizeof(T)*num_elems, cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_second_moment, &second_moment[0], sizeof(T)*num_elems, cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_p, &p[0], sizeof(T)*num_elems, cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);

    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);

    // Launch low-level CUDA kernel
    cuda<T>(stream, num_iter, num_elems, beta_1, beta_2, eps, lr, weight_decay,
        dev_grad, dev_first_moment, dev_second_moment, dev_p);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);

    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&first_moment[0], dev_first_moment, sizeof(T)*num_elems, cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(&second_moment[0], dev_second_moment, sizeof(T)*num_elems, cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(&p[0], dev_p, sizeof(T)*num_elems, cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);

    cuda_err = cudaFree(dev_grad);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_first_moment);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_second_moment);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_p);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
}
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index num_elems, Index num_iter)
{
    if (num_elems == 0)
    {
        return;
    }
    using Y = typename T::repr_t;
    Y eps_check;
    if (std::is_same<T, bf16_t>::value)
    {
        eps_check = 1e-1;
    }
    else if (std::is_same<T, fp16_t>::value)
    {
        eps_check = 1e-2;
    }
    else if (std::is_same<T, fp32_t>::value)
    {
        eps_check = 1e-5;
    }
    else
    {
        eps_check = 1e-8;
    }
    // Parameters
    Scalar beta_1_s = 0.9, beta_2_s = 0.999, eps_s = 1e-8, lr_s = 0.01, weight_decay_s=0.1;
    const Y beta_1{beta_1_s}, beta_2{beta_2_s}, eps{eps_s}, lr{lr_s}, weight_decay{weight_decay_s};

    // Initial data
    std::vector<T> grad(num_elems), p_init(num_elems);
    std::vector<T> first_moment_init(num_elems, T{0.0}), second_moment_init(num_elems, T{0.0});

    for(Index i = 0; i < num_elems; ++i)
    {
        grad[i] = Y(2*i+1-num_elems);
        p_init[i] = Y(num_elems-i);
        if (num_iter > 1) {
            first_moment_init[i] = Y(i+1);
            second_moment_init[i] = Y(num_elems-i);
        }
    }

    // CPU reference implementation
    std::vector<T> p_ref(p_init), first_moment_ref(first_moment_init),
        second_moment_ref(second_moment_init);

    const Y alpha = lr / (Y{1.0} - std::pow(beta_1, num_iter));
    const Y beta = Y{1.0} / std::sqrt(Y{1.0} - std::pow(beta_2, num_iter));
    for(Index i = 0; i < num_elems; ++i)
    {
        Y p_val = static_cast<Y>(p_ref[i]);
        Y grad_val = static_cast<Y>(grad[i]);
        if (weight_decay != 0)
        {
            grad_val += weight_decay * p_val;
        }
        Y f_val, s_val;
        if(num_iter == 1)
        {
            f_val = (Y{1.0} - beta_1) * grad_val;
            s_val = std::sqrt(Y{1.0}-beta_2) * std::fabs(grad_val);
        }
        else
        {
            f_val = static_cast<Y>(first_moment_ref[i]);
            s_val = static_cast<Y>(second_moment_ref[i]);
            f_val = beta_1*f_val + (Y{1.0}-beta_1)*grad_val;
            s_val = std::hypot(std::sqrt(beta_2)*s_val,
                    std::sqrt(Y{1.0}-beta_2)*grad_val);
        }
        first_moment_ref[i] = static_cast<T>(f_val);
        second_moment_ref[i] = static_cast<T>(s_val);
        const Y denom = s_val*beta + eps;
        p_ref[i] = static_cast<T>(p_val - alpha*f_val/denom);
    }

    // Run CPU kernel
    std::vector<T> p_cpu(p_init), first_moment_cpu(first_moment_init),
        second_moment_cpu(second_moment_init);
    std::cout << "Run kernel::adam_step::cpu<" << T::short_name << ">\n";
    cpu<T>(num_iter, num_elems, beta_1_s, beta_2_s, eps_s, lr_s,
        weight_decay_s, &grad[0], &first_moment_cpu[0],
        &second_moment_cpu[0], &p_cpu[0]);

    for(Index i = 0; i < num_elems; ++i)
    {
        Y p_val = Y{p_cpu[i]}, p_ref_val = Y{p_ref[i]};
        if (p_ref_val == Y{0.0})
        {
            TEST_ASSERT(std::abs(p_val) <= eps_check);
        }
        else
        {
            TEST_ASSERT(std::abs(p_val - p_ref_val) / std::abs(p_ref_val) <= eps_check);
        }
    }
    std::cout << "OK: kernel::adam_step::cpu<" << T::short_name << ">\n";

#ifdef NNTILE_USE_CUDA
    // Run CUDA kernel
    std::vector<T> p_cuda(p_init), first_moment_cuda(first_moment_init),
        second_moment_cuda(second_moment_init);
    std::cout << "Run kernel::adam_step::cuda<" << T::short_name << ">\n";
    run_cuda<T>(num_iter, num_elems, beta_1_s, beta_2_s, eps_s, lr_s,
        weight_decay_s, grad, first_moment_cuda, second_moment_cuda, p_cuda);

    for(Index i = 0; i < num_elems; ++i)
    {
        Y p_val = Y{p_cuda[i]}, p_ref_val = Y{p_ref[i]};
        if (p_ref_val == Y{0.0})
        {
            TEST_ASSERT(std::abs(p_val) <= eps_check);
        }
        else
        {
            TEST_ASSERT(std::abs(p_val - p_ref_val) / std::abs(p_ref_val) <= eps_check);
        }
    }
    std::cout << "OK: kernel::adam_step::cuda<" << T::short_name << ">\n";
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    const Index test_nelems[] = {0, 1, 10, 1024};
    const Index test_niter[] = {1, 5};
    for(Index nelems : test_nelems)
    {
        for(Index niter : test_niter)
        {
            validate<fp32_t>(nelems, niter);
            validate<fp64_t>(nelems, niter);
            validate<bf16_t>(nelems, niter);
            validate<fp16_t>(nelems, niter);
        }
    }
}
