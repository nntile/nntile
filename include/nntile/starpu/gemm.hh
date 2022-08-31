/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/gemm.hh
 * GEMM operation for StarPU buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-31
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/constants.hh>
#include <nntile/starpu.hh>
#include <nntile/defs.h>

namespace nntile
{
namespace starpu
{

//! Structure for arguments
template<typename T>
struct gemm_args
{
    TransOp transA;
    TransOp transB;
    Index m;
    Index n;
    Index k;
    T alpha;
    T beta;
};

#ifdef NNTILE_USE_CBLAS
template<typename T>
void gemm_cpu(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CBLAS

#ifdef NNTILE_USE_CUDA
template<typename T>
void gemm_cuda(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CUDA

extern StarpuCodelet gemmNN_codelet_fp32, gemmNN_codelet_fp64,
       gemmNT_codelet_fp32, gemmNT_codelet_fp64,
       gemmTN_codelet_fp32, gemmTN_codelet_fp64,
       gemmTT_codelet_fp32, gemmTT_codelet_fp64;

void gemm_init();

void gemm_restrict_where(uint32_t where);

void gemm_restore_where();

template<typename T>
void gemm(const TransOp &transA, const TransOp &transB, Index m, Index n,
        Index k, T alpha, starpu_data_handle_t A, starpu_data_handle_t B,
        T beta, starpu_data_handle_t C);

} // namespace starpu
} // namespace nntile

