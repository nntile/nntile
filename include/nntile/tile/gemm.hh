#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

template<typename T>
void gemm_async(T alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, T beta, const Tile<T> &C,
        int ndim=1);

extern template
void gemm_async<float>(float alpha, const TransOp &transA,
        const Tile<float> &A, const TransOp &transB, const Tile<float> &B,
        float beta, const Tile<float> &C, int ndim=1);

extern template
void gemm_async<double>(double alpha, const TransOp &transA,
        const Tile<double> &A, const TransOp &transB, const Tile<double> &B,
        double beta, const Tile<double> &C, int ndim=1);

template<typename T>
void gemm(T alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, T beta, const Tile<T> &C,
        int ndim=1)
{
    gemm_async<T>(alpha, transA, A, transB, B, beta, C, ndim);
    starpu_task_wait_for_all();
}

} // namespace nntile

