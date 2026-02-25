/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/conv2d_bwd_input_inplace.cc
 * Backward 2D-Convolution of two tiles in WHCN format to get input grad
 *
 * @version 1.1.0
 * */

#include "nntile/tile/conv2d_bwd_input_inplace.hh"
#include "nntile/starpu/conv2d_bwd_input_inplace.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

template<typename T>
void conv2d_bwd_input_inplace_async(Index src1_m, Index src1_n, Index stride_m,
        Index stride_n, Index src1_channels, Index batch, Index src2_m,
        Index src2_n, Index dilation_m, Index dilation_n, Index dst_channels,
        Index offset_m, Index offset_n, Scalar alpha, const Tile<T> &src1,
        const Tile<T> &src2, Index dst_m, Index dst_n, Scalar beta,
        const Tile<T> &dst)
{
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    src1.mpi_transfer(dst_rank, mpi_rank);
    src2.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        starpu::conv2d_bwd_input_inplace.submit<std::tuple<T>>(src1_m, src1_n,
                stride_m, stride_n, src1_channels, batch, src2_m, src2_n,
                dilation_m, dilation_n, dst_channels, offset_m, offset_n,
                alpha, src1, src2, dst_m, dst_n, beta, dst);
    }
}

template<typename T>
void conv2d_bwd_input_inplace(Index src1_m, Index src1_n, Index stride_m,
        Index stride_n, Index src1_channels, Index batch, Index src2_m,
        Index src2_n, Index dilation_m, Index dilation_n, Index dst_channels,
        Index offset_m, Index offset_n, Scalar alpha, const Tile<T> &src1,
        const Tile<T> &src2, Index dst_m, Index dst_n, Scalar beta,
        const Tile<T> &dst)
{
    conv2d_bwd_input_inplace_async<T>(src1_m, src1_n, stride_m, stride_n,
            src1_channels, batch, src2_m, src2_n, dilation_m, dilation_n,
            dst_channels, offset_m, offset_n, alpha, src1, src2, dst_m, dst_n,
            beta, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void conv2d_bwd_input_inplace_async<bf16_t>(Index src1_m, Index src1_n,
        Index stride_m, Index stride_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index dilation_m, Index dilation_n,
        Index dst_channels, Index offset_m, Index offset_n, Scalar alpha,
        const Tile<bf16_t> &src1, const Tile<bf16_t> &src2, Index dst_m,
        Index dst_n, Scalar beta, const Tile<bf16_t> &dst);

template
void conv2d_bwd_input_inplace_async<fp32_t>(Index src1_m, Index src1_n,
        Index stride_m, Index stride_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index dilation_m, Index dilation_n,
        Index dst_channels, Index offset_m, Index offset_n, Scalar alpha,
        const Tile<fp32_t> &src1, const Tile<fp32_t> &src2, Index dst_m,
        Index dst_n, Scalar beta, const Tile<fp32_t> &dst);

template
void conv2d_bwd_input_inplace_async<fp32_fast_tf32_t>(Index src1_m,
        Index src1_n, Index stride_m, Index stride_n, Index src1_channels,
        Index batch, Index src2_m, Index src2_n, Index dilation_m,
        Index dilation_n, Index dst_channels, Index offset_m, Index offset_n,
        Scalar alpha, const Tile<fp32_fast_tf32_t> &src1,
        const Tile<fp32_fast_tf32_t> &src2, Index dst_m, Index dst_n,
        Scalar beta, const Tile<fp32_fast_tf32_t> &dst);

template
void conv2d_bwd_input_inplace_async<fp32_fast_fp16_t>(Index src1_m,
        Index src1_n, Index stride_m, Index stride_n, Index src1_channels,
        Index batch, Index src2_m, Index src2_n, Index dilation_m,
        Index dilation_n, Index dst_channels, Index offset_m, Index offset_n,
        Scalar alpha, const Tile<fp32_fast_fp16_t> &src1,
        const Tile<fp32_fast_fp16_t> &src2, Index dst_m, Index dst_n,
        Scalar beta, const Tile<fp32_fast_fp16_t> &dst);

template
void conv2d_bwd_input_inplace_async<fp32_fast_bf16_t>(Index src1_m,
        Index src1_n, Index stride_m, Index stride_n, Index src1_channels,
        Index batch, Index src2_m, Index src2_n, Index dilation_m,
        Index dilation_n, Index dst_channels, Index offset_m, Index offset_n,
        Scalar alpha, const Tile<fp32_fast_bf16_t> &src1,
        const Tile<fp32_fast_bf16_t> &src2, Index dst_m, Index dst_n,
        Scalar beta, const Tile<fp32_fast_bf16_t> &dst);

template
void conv2d_bwd_input_inplace_async<fp64_t>(Index src1_m, Index src1_n,
        Index stride_m, Index stride_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index dilation_m, Index dilation_n,
        Index dst_channels, Index offset_m, Index offset_n, Scalar alpha,
        const Tile<fp64_t> &src1, const Tile<fp64_t> &src2, Index dst_m,
        Index dst_n, Scalar beta, const Tile<fp64_t> &dst);

// Explicit instantiation
template
void conv2d_bwd_input_inplace<bf16_t>(Index src1_m, Index src1_n,
        Index stride_m, Index stride_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index dilation_m, Index dilation_n,
        Index dst_channels, Index offset_m, Index offset_n, Scalar alpha,
        const Tile<bf16_t> &src1, const Tile<bf16_t> &src2, Index dst_m,
        Index dst_n, Scalar beta, const Tile<bf16_t> &dst);

template
void conv2d_bwd_input_inplace<fp32_t>(Index src1_m, Index src1_n,
        Index stride_m, Index stride_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index dilation_m, Index dilation_n,
        Index dst_channels, Index offset_m, Index offset_n, Scalar alpha,
        const Tile<fp32_t> &src1, const Tile<fp32_t> &src2, Index dst_m,
        Index dst_n, Scalar beta, const Tile<fp32_t> &dst);

template
void conv2d_bwd_input_inplace<fp32_fast_tf32_t>(Index src1_m, Index src1_n,
        Index stride_m, Index stride_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index dilation_m, Index dilation_n,
        Index dst_channels, Index offset_m, Index offset_n, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &src1, const Tile<fp32_fast_tf32_t> &src2,
        Index dst_m, Index dst_n, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst);

template
void conv2d_bwd_input_inplace<fp32_fast_fp16_t>(Index src1_m, Index src1_n,
        Index stride_m, Index stride_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index dilation_m, Index dilation_n,
        Index dst_channels, Index offset_m, Index offset_n, Scalar alpha,
        const Tile<fp32_fast_fp16_t> &src1, const Tile<fp32_fast_fp16_t> &src2,
        Index dst_m, Index dst_n, Scalar beta,
        const Tile<fp32_fast_fp16_t> &dst);

template
void conv2d_bwd_input_inplace<fp32_fast_bf16_t>(Index src1_m, Index src1_n,
        Index stride_m, Index stride_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index dilation_m, Index dilation_n,
        Index dst_channels, Index offset_m, Index offset_n, Scalar alpha,
        const Tile<fp32_fast_bf16_t> &src1, const Tile<fp32_fast_bf16_t> &src2,
        Index dst_m, Index dst_n, Scalar beta,
        const Tile<fp32_fast_bf16_t> &dst);

template
void conv2d_bwd_input_inplace<fp64_t>(Index src1_m, Index src1_n,
        Index stride_m, Index stride_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index dilation_m, Index dilation_n,
        Index dst_channels, Index offset_m, Index offset_n, Scalar alpha,
        const Tile<fp64_t> &src1, const Tile<fp64_t> &src2, Index dst_m,
        Index dst_n, Scalar beta, const Tile<fp64_t> &dst);

} // namespace nntile::tile
