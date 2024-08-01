/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/conv2d_inplace.cc
 * Forward 2D-Convolution of two tensors in WHCN format
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.0.0
 * */

#include "nntile/tensor/conv2d_inplace.hh"
#include <algorithm>
#include "nntile/starpu/clear.hh"
#include "nntile/starpu/scal_inplace.hh"
#include "nntile/starpu/conv2d_inplace.hh"
#include <unistd.h>

namespace nntile::tensor
{

template <typename T>
void conv2d_inplace_async(Scalar alpha, const Tensor<T> &X,
        const Tensor<T> &C, Scalar beta, const Tensor<T> &Y,
        Index padding_m, Index padding_n)
/*! Forward 2D convolution of two tensors in WHCN format
 *
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * Current implementation requires C to contain only a single tile
 *
 * The following operation is performed:
 *      `Y` = `alpha`*`f(X, C)` + `beta`*`Y`,
 * where `f` operation does the following:
 *      `f[i,j,k,b]` = \sum_l \sum_m \sum_n `X[m,n,l,b]`
 *      * `C[m + offset_m - i,n + offset_n - j,l,k]`
 *
 * Generally, `X` represents input of `Conv2d` layer, `C` represents
 * kernel of `Conv2d` layer and `Y` represents output of `Conv2d` layer.
 *
 * X must be of shape (W_in, H_in, C_in, N)
 * X must have basetile (W_in_tile, H_in_tile, C_in, N_tile)
 * C must be of shape (K_x, K_y, C_in, C_out)
 * C must have basetile (K_x, K_y, C_in, C_out)
 * Y must be of shape (W_out, H_out, C_out, N)
 * with W_out=W_in-K_x+2padding_m+1
 * and H_out=H_in-K_y+2padding_n+1
 * Y must have basetile (W_out_tile, H_out_tile, C_out, N_tile)
 *
 * @param[in] X: Input tensor, that is usually a source for Conv2d.
 * @param[in] C: Input tensor, that is usually a kernel for Conv2d.
 * @param[inout] Y: Resulting tensor
 * @param[in] padding_m: Padding on the second axis of the input
 * @param[in] padding_n: Padding on the first axis of the input
 * */
{
    // Check dimensions
    if(4 != X.ndim)
    {
        throw std::runtime_error("4 != X.ndim");
    }
    if(4 != Y.ndim)
    {
        throw std::runtime_error("4 != Y.ndim");
    }
    // Check shapes of tensors
    if(Y.shape[0] != X.shape[0] - C.shape[0] + 1 + 2*padding_m)
    {
        throw std::runtime_error("Y.shape[0] != X.shape[0] - "
                                 "C.shape[0] + 1 + 2*padding_n");
    }
    if(Y.shape[1] != X.shape[1] - C.shape[1] + 1 + 2*padding_n)
    {
        throw std::runtime_error("Y.shape[1] != X.shape[1] - "
                                 "C.shape[1] + 1 + 2*padding_m");
    }
    if(X.shape[2] != C.shape[2])
    {
        throw std::runtime_error("X.shape[2] != C.shape[2]");
    }
    if(Y.shape[2] != C.shape[3])
    {
        throw std::runtime_error("Y.shape[2] != C.shape[3]");
    }
    if(X.shape[3] != Y.shape[3])
    {
        throw std::runtime_error("X.shape[3] != Y.shape[3]");
    }
    // Check base tiles
    if(X.basetile_shape[2] != X.shape[2])
    {
        throw std::runtime_error("X.basetile_shape[2] != X.shape[2]");
    }
    if(Y.basetile_shape[2] != Y.shape[2])
    {
        throw std::runtime_error("Y.basetile_shape[2] != Y.shape[2]");
    }
    if(X.basetile_shape[3] != Y.basetile_shape[3])
    {
        throw std::runtime_error("X.basetile_shape[3] != "
                "Y.basetile_shape[3]");
    }
    if(C.shape != C.basetile_shape)
    {
        throw std::runtime_error("C.shape != C.basetile_shape");
    }

    // Loop through output tiles
    for(Index i = 0; i < Y.grid.nelems; ++i)
    {
        auto Y_tile_index = Y.grid.linear_to_index(i);
        auto Y_tile_traits = Y.get_tile_traits(i);
        auto Y_tile_handle = Y.get_tile_handle(i);
        // Get start and end coordinates of dst tile within Y tensor
        Index Y_start_m = Y_tile_index[0] * Y.basetile_shape[0];
        Index Y_end_m = Y_start_m + Y_tile_traits.shape[0];
        Index Y_start_n = Y_tile_index[1] * Y.basetile_shape[1];
        Index Y_end_n = Y_start_n + Y_tile_traits.shape[1];
        // Get X start and end coordinates that interact with Y through
        // provided kernel
        Index X_start_m = Y_start_m - padding_m;
        Index X_end_m = Y_end_m - padding_m + C.shape[0];
        Index X_start_n = Y_start_n - padding_n;
        Index X_end_n = Y_end_n - padding_n + C.shape[1];
        // Get X tile coordinates that interact with Y tile
        Index X_start_tile_m = X_start_m / X.basetile_shape[0];
        Index X_end_tile_m = (X_end_m-1) / X.basetile_shape[0] + 1;
        Index X_start_tile_n = X_start_n / X.basetile_shape[1];
        Index X_end_tile_n = (X_end_n-1) / X.basetile_shape[1] + 1;
        // Case of big padding: Y tile does not require any convolution
        if(X_end_tile_m <= 0 or X_start_tile_m >= X.grid.shape[0]
                or X_end_tile_n <= 0 or X_start_tile_n >= X.grid.shape[1])
        {
            // Clear if beta is zero
            if(beta == 0.0)
            {
                starpu::clear::submit(Y_tile_handle);
            }
            // Scale inplace if beta is neither zero nor one
            else if(beta != 1.0)
            {
                starpu::scal_inplace::submit<T>(Y_tile_traits.nelems, beta,
                        Y_tile_handle);
            }
            // Do nothing if beta is one
            // Cycle to the next Y tile
            continue;
        }
        // Loop through corresponding X tiles
        std::vector<Index> X_tile_index(Y_tile_index);
        Index start_m = std::max(X_start_tile_m, Index(0));
        Index end_m = std::min(X_end_tile_m, X.grid.shape[0]);
        Index start_n = std::max(X_start_tile_n, Index(0));
        Index end_n = std::min(X_end_tile_n, X.grid.shape[1]);
        Scalar Y_tile_beta = beta;
        for(Index X_i = start_m; X_i < end_m; ++X_i)
        {
            X_tile_index[0] = X_i;
            for(Index X_j = start_n; X_j < end_n; ++X_j)
            {
                X_tile_index[1] = X_j;
                auto X_tile_traits = X.get_tile_traits(X_tile_index);
                auto X_tile_handle = X.get_tile_handle(X_tile_index);
                Index offset_m = X_i*X.basetile_shape[0] - Y_start_m
                    + padding_m;
                Index offset_n = X_j*X.basetile_shape[1] - Y_start_n
                    + padding_n;
                starpu::conv2d_inplace::submit<T>(X_tile_traits.shape[0],
                        X_tile_traits.shape[1], X_tile_traits.shape[2],
                        X_tile_traits.shape[3], C.shape[0],
                        C.shape[1], Y_tile_traits.shape[2], offset_m,
                        offset_n, alpha, X_tile_handle, C.get_tile_handle(0),
                        Y_tile_traits.shape[0], Y_tile_traits.shape[1],
                        Y_tile_beta, Y_tile_handle);
                Y_tile_beta = 1.0;
            }
        }
    }
}

template <typename T>
void conv2d_inplace(Scalar alpha, const Tensor<T> &X, const Tensor<T> &C,
        Scalar beta, const Tensor<T> &Y, Index padding_m, Index padding_n)
/*! Blocking version of conv2d_inplace_async<T>.
 *
 * @param[in] X: Input tensor, that is usually a source for Conv2d.
 * @param[in] C: Input tensor, that is usually a kernel for Conv2d.
 * @param[inout] Y: Resulting tensor
 * @param[in] padding_m: Padding on the second axis of the input
 * @param[in] padding_n: Padding on the first axis of the input
 * */
{
    conv2d_inplace_async<T>(alpha, X, C, beta, Y, padding_m, padding_n);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template void conv2d_inplace_async<bf16_t>(Scalar alpha,
        const Tensor<bf16_t> &X, const Tensor<bf16_t> &C, Scalar beta,
        const Tensor<bf16_t> &Y, Index padding_m, Index padding_n);

template void conv2d_inplace_async<fp32_t>(Scalar alpha,
        const Tensor<fp32_t> &X, const Tensor<fp32_t> &C, Scalar beta,
        const Tensor<fp32_t> &Y, Index padding_m, Index padding_n);

template void conv2d_inplace_async<fp32_fast_tf32_t>(Scalar alpha,
        const Tensor<fp32_fast_tf32_t> &X,
        const Tensor<fp32_fast_tf32_t> &C, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &Y, Index padding_m, Index padding_n);

template void conv2d_inplace_async<fp64_t>(Scalar alpha,
        const Tensor<fp64_t> &X, const Tensor<fp64_t> &C, Scalar beta,
        const Tensor<fp64_t> &Y, Index padding_m, Index padding_n);

// Explicit instantiation of template
template void conv2d_inplace<bf16_t>(Scalar alpha,
        const Tensor<bf16_t> &X, const Tensor<bf16_t> &C, Scalar beta,
        const Tensor<bf16_t> &Y, Index padding_m, Index padding_n);

template void conv2d_inplace<fp32_t>(Scalar alpha,
        const Tensor<fp32_t> &X, const Tensor<fp32_t> &C, Scalar beta,
        const Tensor<fp32_t> &Y, Index padding_m, Index padding_n);

template void conv2d_inplace<fp32_fast_tf32_t>(Scalar alpha,
        const Tensor<fp32_fast_tf32_t> &X,
        const Tensor<fp32_fast_tf32_t> &C, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &Y, Index padding_m, Index padding_n);

template void conv2d_inplace<fp64_t>(Scalar alpha,
        const Tensor<fp64_t> &X, const Tensor<fp64_t> &C, Scalar beta,
        const Tensor<fp64_t> &Y, Index padding_m, Index padding_n);

} // namespace nntile::tensor
