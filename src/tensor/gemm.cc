/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/gemm.cc
 * GEMM operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-12-01
 * */

#include "nntile/tensor/gemm.hh"
#include "nntile/starpu/gemm.hh"

namespace nntile
{
namespace tensor
{

//! Check if dimensionalities of tensors match gemm
static inline void gemm_check_ndim(const TensorTraits &A,
        const TensorTraits &B, const TensorTraits &C, Index ndim)
{
    // Check if ndim is negative since it will be converted to Index
    if(ndim <= 0)
    {
        throw std::runtime_error("ndim <= 0");
    }
    if(A.ndim < ndim)
    {
        throw std::runtime_error("A.ndim < ndim");
    }
    if(B.ndim < ndim)
    {
        throw std::runtime_error("B.ndim < ndim");
    }
    if(A.ndim + B.ndim != C.ndim + 2*ndim)
    {
        throw std::runtime_error("A.ndim + B.ndim != C.ndim + 2*ndim");
    }
}

//! Check if shapes of tensors A and B match gemm
static inline void gemm_check_A_B(const TensorTraits &A,
        const TensorTraits &B, Index ndim=1)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[A.ndim-ndim+i] != B.shape[i])
        {
            throw std::runtime_error("A.shape[A.ndim-ndim:A.ndim] != "
                    "B.shape[0:ndim]");
        }
        if(A.basetile_shape[A.ndim-ndim+i] != B.basetile_shape[i])
        {
            throw std::runtime_error("A.basetile_shape[A.ndim-ndim:A.ndim] != "
                    "B.basetile_shape[0:ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and B match gemm
static inline void gemm_check_AT_B(const TensorTraits &A,
        const TensorTraits &B, Index ndim=1)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[i] != B.shape[i])
        {
            throw std::runtime_error("A.shape[0:ndim] != B.shape[0:ndim]");
        }
        if(A.basetile_shape[i] != B.basetile_shape[i])
        {
            throw std::runtime_error("A.basetile_shape[0:ndim] != "
                    "B.basetile_shape[0:ndim]");
        }
    }
}

//! Check if shapes of tensors A and B^T match gemm
static inline void gemm_check_A_BT(const TensorTraits &A,
        const TensorTraits &B, Index ndim=1)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[A.ndim-ndim+i] != B.shape[B.ndim-ndim+i])
        {
            throw std::runtime_error("A.shape[A.ndim-ndim:A.ndim] != "
                    "B.shape[B.ndim-ndim:B.ndim]");
        }
        if(A.basetile_shape[A.ndim-ndim+i] != B.basetile_shape[B.ndim-ndim+i])
        {
            throw std::runtime_error("A.basetile_shape[A.ndim-ndim:A.ndim] != "
                    "B.shape[B.ndim-ndim:B.ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and B^T match gemm
static inline void gemm_check_AT_BT(const TensorTraits &A,
        const TensorTraits &B, Index ndim=1)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[i] != B.shape[B.ndim-ndim+i])
        {
            throw std::runtime_error("A.shape[0:ndim] != "
                    "B.shape[B.ndim-ndim:B.ndim]");
        }
        if(A.basetile_shape[i] != B.basetile_shape[B.ndim-ndim+i])
        {
            throw std::runtime_error("A.basetile_shape[0:ndim] != "
                    "B.basetile_shape[B.ndim-ndim:B.ndim]");
        }
    }
}

//! Check if shapes of tensors op(A) and op(B) match gemm
static inline void gemm_check_opA_opB(const TransOp &transA,
        const TensorTraits &A, const TransOp &transB, const TensorTraits &B,
        Index ndim=1)
{
    switch(transB.value)
    {
        case TransOp::NoTrans:
            switch(transA.value)
            {
                case TransOp::NoTrans:
                    gemm_check_A_B(A, B, ndim);
                    break;
                case TransOp::Trans:
                    gemm_check_AT_B(A, B, ndim);
                    break;
                default:
                    throw std::runtime_error("Wrong value of transA");
            }
            break;
        case TransOp::Trans:
            switch(transA.value)
            {
                case TransOp::NoTrans:
                    gemm_check_A_BT(A, B, ndim);
                    break;
                case TransOp::Trans:
                    gemm_check_AT_BT(A, B, ndim);
                    break;
                default:
                    throw std::runtime_error("Wrong value of transA");
            }
            break;
        default:
            throw std::runtime_error("Wrong value of transB");
    }
}

//! Check if shapes of tensors A and C match gemm
static inline void gemm_check_A_C(const TensorTraits &A,
        const TensorTraits &C, Index ndim=1)
{
    for(Index i = 0; i < A.ndim-ndim; ++i)
    {
        if(A.shape[i] != C.shape[i])
        {
            throw std::runtime_error("A.shape[0:A.ndim-ndim] != "
                    "C.shape[0:A.ndim-ndim]");
        }
        if(A.basetile_shape[i] != C.basetile_shape[i])
        {
            throw std::runtime_error("A.basetile_shape[0:A.ndim-ndim] != "
                    "C.basetile_shape[0:A.ndim-ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and C match gemm
static inline void gemm_check_AT_C(const TensorTraits &A,
        const TensorTraits &C, Index ndim=1)
{
    for(Index i = ndim; i < A.ndim; ++i)
    {
        if(A.shape[i] != C.shape[i-ndim])
        {
            throw std::runtime_error("A.shape[ndim:A.ndim] != "
                    "C.shape[0:A.ndim-ndim]");
        }
        if(A.basetile_shape[i] != C.basetile_shape[i-ndim])
        {
            throw std::runtime_error("A.basetile_shape[ndim:A.ndim] != "
                    "C.basetile_shape[0:A.ndim-ndim]");
        }
    }
}

//! Check if shapes of tensors op(A) and C match gemm
static inline void gemm_check_opA_C(const TransOp &transA,
        const TensorTraits &A, const TensorTraits &C, Index ndim=1)
{
    switch(transA.value)
    {
        case TransOp::NoTrans:
            gemm_check_A_C(A, C, ndim);
            break;
        case TransOp::Trans:
            gemm_check_AT_C(A, C, ndim);
            break;
        // This parameter was already checked in gemm_check_opA_opB
    }
}

//! Check if shapes of tensors B and C match gemm
static inline void gemm_check_B_C(const TensorTraits &B,
        const TensorTraits &C, Index ndim=1)
{
    for(Index i = ndim; i < B.ndim; ++i)
    {
        if(B.shape[i] != C.shape[C.ndim-B.ndim+i])
        {
            throw std::runtime_error("B.shape[ndim:B.ndim] != "
                    "C.shape[C.ndim-B.ndim+ndim:C.ndim]");
        }
        if(B.basetile_shape[i] != C.basetile_shape[C.ndim-B.ndim+i])
        {
            throw std::runtime_error("B.basetile_shape[ndim:B.ndim] != "
                    "C.basetile_shape[C.ndim-B.ndim+ndim:C.ndim]");
        }
    }
}

//! Check if shapes of tensors B^T and C match gemm
static inline void gemm_check_BT_C(const TensorTraits &B,
        const TensorTraits &C, Index ndim=1)
{
    for(Index i = 0; i < B.ndim-ndim; ++i)
    {
        if(B.shape[i] != C.shape[C.ndim-B.ndim+ndim+i])
        {
            throw std::runtime_error("B.shape[0:B.ndim-ndim] != "
                    "C.shape[C.ndim-B.ndim+ndim:C.ndim]");
        }
        if(B.basetile_shape[i] != C.basetile_shape[C.ndim-B.ndim+ndim+i])
        {
            throw std::runtime_error("B.basetile_shape[0:B.ndim-ndim] != "
                    "C.basetile_shape[C.ndim-B.ndim+ndim:C.ndim]");
        }
    }
}

//! Check if shapes of tensors op(B) and C match gemm
static inline void gemm_check_opB_C(const TransOp &transB,
        const TensorTraits &B, const TensorTraits &C, Index ndim=1)
{
    switch(transB.value)
    {
        case TransOp::NoTrans:
            gemm_check_B_C(B, C, ndim);
            break;
        case TransOp::Trans:
            gemm_check_BT_C(B, C, ndim);
            break;
        // This parameter was already checked in gemm_check_opA_opB
    }
}

//! Check if tensors match gemm
void gemm_check(const TransOp &transA, const TensorTraits &A,
        const TransOp &transB, const TensorTraits &B, const TensorTraits &C,
        Index ndim)
{
    // Check if dimensionalities match
    gemm_check_ndim(A, B, C, ndim);
    // Check if shapes of A and B match
    gemm_check_opA_opB(transA, A, transB, B, ndim);
    // Check if shapes of A and C match
    gemm_check_opA_C(transA, A, C, ndim);
    // Check if shapes of B and C match
    gemm_check_opB_C(transB, B, C, ndim);
}

//! Asynchronous version of tensor-wise gemm operation
/*! Matrix multiplication for tensors, which are virtually reshaped
 *
 * @param[in] alpha: Alpha multiplier
 * @param[in] transA: Transposition flag for the tensor A
 * @param[in] A: Input tensor A
 * @param[in] transB: Transposition flag for the tensor B
 * @param[in] B: Input tensor B
 * @param[in] beta: Beta multiplier
 * @param[inout] C: Output tensor C
 * @param[in] ndim: Number of dimensions used in gemm contraction
 * */
template<typename T>
void gemm_async(T alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, T beta, const Tensor<T> &C,
        Index ndim)
{
    // Check inputs (throw exception in case of an error)
    gemm_check(transA, A, transB, B, C, ndim);
    // Sizes of A, B and C as simple matrices (grids of tiles) for gemm
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    constexpr T one = 1;
    Index m = C.grid.matrix_shape[A.ndim-ndim][0];
    Index n = C.grid.matrix_shape[A.ndim-ndim][1];
    Index k;
    std::array<Index, 2> opA_stride, opB_stride;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            k = A.grid.matrix_shape[A.ndim-ndim][1];
            opA_stride = {1, m};
            break;
        case TransOp::Trans:
            k = A.grid.matrix_shape[ndim][0];
            opA_stride = {k, 1};
            break;
    }
    switch(transB.value)
    {
        case TransOp::NoTrans:
            opB_stride = {1, k};
            break;
        case TransOp::Trans:
            opB_stride = {n, 1};
            break;
    }
    // All per-tile starpu gemm calls shall appear here
    for(Index j = 0; j < n; ++j)
    {
        for(Index i = 0; i < m; ++i)
        {
            Index C_tile_offset = j*m + i;
            auto C_tile_handle = C.get_tile_handle(C_tile_offset);
            auto C_tile_traits = C.get_tile_traits(C_tile_offset);
            int C_tile_rank = C_tile_handle.mpi_get_rank();
            Index tile_m = C_tile_traits.matrix_shape[A.ndim-ndim][0];
            Index tile_n = C_tile_traits.matrix_shape[A.ndim-ndim][1];
            // initialize C(i,j) = a*opA(i,0)*opB(0,j) + b*C(i,j)
            Index A_tile_offset = opA_stride[0] * i;
            Index B_tile_offset = opB_stride[1] * j;
            auto A_first_tile_handle = A.get_tile_handle(A_tile_offset);
            auto B_first_tile_handle = B.get_tile_handle(B_tile_offset);
            int A_first_tile_rank = A_first_tile_handle.mpi_get_rank();
            int B_first_tile_rank = B_first_tile_handle.mpi_get_rank();
            // Transfer first tile A on node with tile C
            A_first_tile_handle.mpi_transfer(C_tile_rank, mpi_rank);
            // Transfer first tile B on node with tile C
            B_first_tile_handle.mpi_transfer(C_tile_rank, mpi_rank);
            // Execute on node with tile C
            if(mpi_rank == C_tile_rank)
            {
                Index tile_k;
                auto A_first_tile_traits = A.get_tile_traits(A_tile_offset);
                switch(transA.value)
                {
                    case TransOp::NoTrans:
                        tile_k = A_first_tile_traits.matrix_shape[
                            A.ndim-ndim][1];
                        break;
                        // This parameter was already checked
                        //case TransOp::Trans:
                    default:
                        tile_k = A_first_tile_traits.matrix_shape[ndim][0];
                        break;
                }
                starpu::gemm::submit<T>(transA, transB, tile_m, tile_n, tile_k,
                        alpha, A_first_tile_handle, B_first_tile_handle, beta,
                        C_tile_handle);
            }
            // all other l>0
            for(Index l = 1; l < k; ++l)
            {
                // accumulate C(i,j) = a*opA(i,l)*opB(l,j) + C(i,j)
                A_tile_offset += opA_stride[1];
                B_tile_offset += opB_stride[0];
                auto A_tile_handle = A.get_tile_handle(A_tile_offset);
                auto B_tile_handle = B.get_tile_handle(B_tile_offset);
                int A_tile_rank = A_tile_handle.mpi_get_rank();
                int B_tile_rank = B_tile_handle.mpi_get_rank();
                // Transfer tile A on node with tile C
                A_tile_handle.mpi_transfer(C_tile_rank, mpi_rank);
                // Transfer tile B on node with tile C
                B_tile_handle.mpi_transfer(C_tile_rank, mpi_rank);
                // Execute on node with tile C
                if(mpi_rank == C_tile_rank)
                {
                    Index tile_k;
                    auto A_tile_traits = A.get_tile_traits(A_tile_offset);
                    switch(transA.value)
                    {
                        case TransOp::NoTrans:
                            tile_k = A_tile_traits.matrix_shape[
                                A.ndim-ndim][1];
                            break;
                            // This parameter was already checked
                            //case TransOp::Trans:
                        default:
                            tile_k = A_tile_traits.matrix_shape[ndim][0];
                            break;
                    }
                    starpu::gemm::submit<T>(transA, transB, tile_m, tile_n,
                            tile_k, alpha, A_tile_handle, B_tile_handle, one,
                            C_tile_handle);
                }
            }
            // Flush cache for the output tile on every node
            C_tile_handle.mpi_flush();
        }
    }
}

//! Blocking version of tensor-wise gemm operation
/*! Matrix multiplication for tensors, which are virtually reshaped
 *
 * @param[in] alpha: Alpha multiplier
 * @param[in] transA: Transposition flag for the tensor A
 * @param[in] A: Input tensor A
 * @param[in] transB: Transposition flag for the tensor B
 * @param[in] B: Input tensor B
 * @param[in] beta: Beta multiplier
 * @param[inout] C: Output tensor C
 * @param[in] ndim: Number of dimensions used in gemm contraction
 * */
template<typename T>
void gemm(T alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, T beta, const Tensor<T> &C,
        Index ndim)
{
    gemm_async<T>(alpha, transA, A, transB, B, beta, C, ndim);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void gemm_async<fp32_t>(fp32_t alpha, const TransOp &transA,
        const Tensor<fp32_t> &A,
        const TransOp &transB, const Tensor<fp32_t> &B, fp32_t beta,
        const Tensor<fp32_t> &C, Index ndim);

template
void gemm_async<fp64_t>(fp64_t alpha, const TransOp &transA,
        const Tensor<fp64_t> &A,
        const TransOp &transB, const Tensor<fp64_t> &B, fp64_t beta,
        const Tensor<fp64_t> &C, Index ndim);

// Explicit instantiation
template
void gemm<fp32_t>(fp32_t alpha, const TransOp &transA, const Tensor<fp32_t> &A,
        const TransOp &transB, const Tensor<fp32_t> &B, fp32_t beta,
        const Tensor<fp32_t> &C, Index ndim);

template
void gemm<fp64_t>(fp64_t alpha, const TransOp &transA, const Tensor<fp64_t> &A,
        const TransOp &transB, const Tensor<fp64_t> &B, fp64_t beta,
        const Tensor<fp64_t> &C, Index ndim);

} // namespace tensor
} // namespace nntile

