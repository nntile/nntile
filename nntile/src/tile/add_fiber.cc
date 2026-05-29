/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/add_fiber.cc
 * Tile wrappers for addition of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

#include "nntile/tile/add_fiber.hh"
#include "nntile/starpu/add_fiber.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

template<typename T>
void add_fiber_async(Scalar alpha, const Tile<T> &src1, Scalar beta, const Tile<T> &src2,
                    const Tile<T> &dst, Index axis, Index batch_ndim)
//! Tile<T> addition of a tensor and a broadcasted fiber
/*! Reshapes input tensor and fiber into 3-dimensional and 1-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j,b] = beta*src2[i,l,j,b] + alpha*src1[l,b]
 *
 * @param[in] alpha: Scalar factor for src1
 * @param[in] src: Input fiber, that is reshaped into 1D array
 * @param[in] beta: Scaling factor for src2
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    // Check dimensions
    if(src1.ndim != batch_ndim+1)
    {
        throw std::runtime_error("src.ndim != batch_ndim+1");
    }
    if(src2.ndim != dst.ndim)
    {
        throw std::runtime_error("src2.ndim != dst.ndim");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= dst.ndim-batch_ndim)
    {
        throw std::runtime_error("axis >= dst.ndim-batch_ndim");
    }
    // Check shapes of tiles
    if(src1.shape[0] != dst.shape[axis])
    {
        throw std::runtime_error("src1.shape[0] != dst.shape[axis]");
    }
    if(src2.shape != dst.shape)
    {
        throw std::runtime_error("src2.shape != dst.shape");
    }
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(src1.shape[i+1] != dst.shape[dst.ndim-batch_ndim+i])
        {
            throw std::runtime_error("src1.shape[i+1] != "
                    "dst.shape[dst.ndim-batch_ndim+i]");
        }
    }

    // Reshape inputs for simplicity: src -> (k,batch), dst -> (m,k,n,batch)
    Index m, n, k, batch;
    batch = src1.matrix_shape[1][1];
    m = dst.stride[axis];
    n = dst.matrix_shape[axis+1][1] / batch;
    k = dst.shape[axis];
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    src1.mpi_transfer(dst_rank, mpi_rank);
    src2.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        // Insert corresponding task
        starpu::add_fiber.submit<std::tuple<T>>(m, n, k, batch, alpha, src1,
                beta, src2, dst);
    }
}

template<typename T>
void add_fiber(Scalar alpha, const Tile<T> &src1, Scalar beta, const Tile<T> &src2,
        const Tile<T> &dst, Index axis, Index batch_ndim)
//! Tile<T> addition of a tensor and a broadcasted fiber
/*! Blocking version of add_fiber_async<T>.
 * Reshapes input tensor and fiber into 3-dimensional and 1-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j,b] = beta*src2[i,l,j,b] + alpha*src1[l,b]
 *
 * @param[in] alpha: Scalar factor for src1
 * @param[in] src: Input fiber, that is reshaped into 1D array
 * @param[in] beta: Scaling factor for src2
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    add_fiber_async<T>(alpha, src1, beta, src2, dst, axis, batch_ndim);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void add_fiber_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src1,
        Scalar beta, const Tile<fp32_t> &src2, const Tile<fp32_t> &dst,
        Index axis, Index batch_ndim);

template
void add_fiber_async<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src1,
        Scalar beta, const Tile<fp32_fast_tf32_t> &src2, const Tile<fp32_fast_tf32_t> &dst,
        Index axis, Index batch_ndim);

template
void add_fiber_async<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src1,
        Scalar beta, const Tile<fp32_fast_fp16_t> &src2, const Tile<fp32_fast_fp16_t> &dst,
        Index axis, Index batch_ndim);

template
void add_fiber_async<fp32_fast_bf16_t>(Scalar alpha, const Tile<fp32_fast_bf16_t> &src1,
        Scalar beta, const Tile<fp32_fast_bf16_t> &src2, const Tile<fp32_fast_bf16_t> &dst,
        Index axis, Index batch_ndim);

template
void add_fiber_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src1,
        Scalar beta, const Tile<fp64_t> &src2, const Tile<fp64_t> &dst,
        Index axis, Index batch_ndim);

template
void add_fiber_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &src1,
        Scalar beta, const Tile<bf16_t> &src2, const Tile<bf16_t> &dst,
        Index axis, Index batch_ndim);

// Explicit instantiation of template
template
void add_fiber<fp32_t>(Scalar alpha, const Tile<fp32_t> &src1,
        Scalar beta, const Tile<fp32_t> &src2, const Tile<fp32_t> &dst,
        Index axis, Index batch_ndim);

template
void add_fiber<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src1,
        Scalar beta, const Tile<fp32_fast_tf32_t> &src2, const Tile<fp32_fast_tf32_t> &dst,
        Index axis, Index batch_ndim);

template
void add_fiber<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src1,
        Scalar beta, const Tile<fp32_fast_fp16_t> &src2, const Tile<fp32_fast_fp16_t> &dst,
        Index axis, Index batch_ndim);

template
void add_fiber<fp32_fast_bf16_t>(Scalar alpha, const Tile<fp32_fast_bf16_t> &src1,
        Scalar beta, const Tile<fp32_fast_bf16_t> &src2, const Tile<fp32_fast_bf16_t> &dst,
        Index axis, Index batch_ndim);

template
void add_fiber<fp64_t>(Scalar alpha, const Tile<fp64_t> &src1,
        Scalar beta, const Tile<fp64_t> &src2, const Tile<fp64_t> &dst,
        Index axis, Index batch_ndim);

template
void add_fiber<bf16_t>(Scalar alpha, const Tile<bf16_t> &src1,
        Scalar beta, const Tile<bf16_t> &src2, const Tile<bf16_t> &dst,
        Index axis, Index batch_ndim);

} // namespace nntile::tile
