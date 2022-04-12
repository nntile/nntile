#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{

template<typename T>
void gemm_async(T alpha,
        const TransOp &transA,
        const Tensor<T> &A,
        const TransOp &transB,
        const Tensor<T> &B,
        T beta,
        const Tensor<T> &C,
        int ndim=1);

extern template
void gemm_async(float alpha,
        const TransOp &transA,
        const Tensor<float> &A,
        const TransOp &transB,
        const Tensor<float> &B,
        float beta,
        const Tensor<float> &C,
        int ndim=1);

extern template
void gemm_async(double alpha,
        const TransOp &transA,
        const Tensor<double> &A,
        const TransOp &transB,
        const Tensor<double> &B,
        double beta,
        const Tensor<double> &C,
        int ndim=1);

template<typename T>
void gemm(T alpha,
        const TransOp &transA,
        const Tensor<T> &A,
        const TransOp &transB,
        const Tensor<T> &B,
        T beta,
        const Tensor<T> &C,
        int ndim=1)
{
    gemm_async<T>(alpha, transA, A, transB, B, beta, C, ndim);
    starpu_task_wait_for_all();
}

} // namespace nntile

