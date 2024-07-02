/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/conv2d.cc
 * Tensor wrappers for 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-03
 * */

#include "nntile/tensor/conv2d.hh"
#include "nntile/starpu/clear.hh"
#include "nntile/starpu/conv2d.hh"
#include <unistd.h>
namespace nntile
{
namespace tensor
{

template <typename T>
void conv2d_async(const Tensor<T> &src, const Tensor<T> &kernel,
                  const Tensor<T> &dst)
//! Tensor<T> 2D-Convolution between 2 matrices
/*! Reshapes input tensors into 2-dimensional arrays
 * and performs the 2D-Convolution
 *
 * @param[in] src: Input tensor, that is reshaped into 2D array
 * @param[in] kernel: Input tensor, that is reshaped into 2D array
 * @param[out] dst: Resulting tensor, that is reshaped into 2D array
 * */
{
    Index axis = 0;
    Index batch_ndim = src.ndim - 3;
    Index in_channels = src.grid.matrix_shape[src.ndim - batch_ndim - 1][1] /
                        src.grid.matrix_shape[src.ndim - batch_ndim][1];
    Index out_channels = dst.grid.matrix_shape[src.ndim - batch_ndim - 1][1] /
                         src.grid.matrix_shape[src.ndim - batch_ndim][1];
    // Check dimensions
    if(2 > src.ndim)
    {
        throw std::runtime_error("2 > src.ndim");
    }
    if(src.ndim != dst.ndim)
    {
        throw std::runtime_error("src.ndim != dst.ndim");
    }
    // Check shapes of tensors
    for(Index i = 0; i < 2; ++i)
    {
        if(dst.shape[i] != src.shape[i] + kernel.shape[i] - 1)
        {
            throw std::runtime_error(
                "dst.shape[i] != src.shape[i] + kernel.shape[i] - 1");
        }
    }

    Index batch = src.grid.matrix_shape[src.ndim - batch_ndim][1];
    Index tile_batch =
        src.get_tile_traits(0).matrix_shape[src.ndim - batch_ndim][1];
    Index kernel_batch = kernel.grid.matrix_shape[src.ndim - batch_ndim][1];
    Index kernel_tile_batch =
        kernel.get_tile_traits(0).matrix_shape[src.ndim - batch_ndim][1];

    // Getting sizes
    Index src_m = src.grid.matrix_shape[1][0];
    Index src_n = src.grid.matrix_shape[2][0] / src_m;
    Index src_tile_m = src.get_tile_traits(0).matrix_shape[1][0];
    Index src_tile_n = src.get_tile_traits(0).matrix_shape[2][0] / src_tile_m;

    Index kernel_m = kernel.grid.matrix_shape[1][0];
    Index kernel_n = kernel.grid.matrix_shape[2][0] / kernel_m;
    Index kernel_tile_m = kernel.get_tile_traits(0).matrix_shape[1][0];
    Index kernel_tile_n =
        kernel.get_tile_traits(0).matrix_shape[2][0] / kernel_tile_m;

    Index dst_m = dst.grid.matrix_shape[1][0];
    Index dst_n = dst.grid.matrix_shape[2][0] / dst_m;
    Index dst_tile_m = dst.get_tile_traits(0).matrix_shape[1][0];
    Index dst_tile_n = dst.get_tile_traits(0).matrix_shape[2][0] / dst_tile_m;

    for(Index dst_index = 0; dst_index < dst.grid.nelems; ++dst_index)
    {
        auto dst_tile_handle = dst.get_tile_handle(dst_index);
        starpu::clear::submit(dst_tile_handle);
    }

    for(Index b = 0; b < batch; ++b)
    {
        for(Index oc = 0; oc < out_channels; ++oc)
        {
            Index tile_oc_current =
                dst.get_tile_traits(oc * dst_n * dst_m).matrix_shape[3][0] /
                dst.get_tile_traits(oc * dst_n * dst_m).matrix_shape[2][0];
            for(Index ic = 0; ic < in_channels; ++ic)
            {
                Index tile_ic_current =
                    src.get_tile_traits(ic * src_n * src_m).matrix_shape[3][0] /
                    src.get_tile_traits(ic * src_n * src_m).matrix_shape[2][0];
                for(Index src_i = 0; src_i < src_n; ++src_i)
                {
                    for(Index src_j = 0; src_j < src_m; ++src_j)
                    {
                        Index src_index = src_j + src_i * src_m +
                                          ic * src_n * src_m +
                                          b * src_n * src_m * in_channels;
                        auto src_tile_handle = src.get_tile_handle(src_index);

                        Index tile_batch_current =
                            src.get_tile_traits(src_index).matrix_shape[3][1];

                        Index src_tile_m_current =
                            src.get_tile_traits(src_index).matrix_shape[1][0];
                        Index src_tile_n_current =
                            src.get_tile_traits(src_index).matrix_shape[2][0] /
                            src_tile_m_current;
                        Index src_offset_n = src_i * src_tile_n;
                        Index src_offset_m = src_j * src_tile_m;

                        for(Index kernel_i = 0; kernel_i < kernel_n; ++kernel_i)
                        {
                            for(Index kernel_j = 0; kernel_j < kernel_m;
                                ++kernel_j)
                            {
                                Index kernel_index =
                                    kernel_j + kernel_i * kernel_m +
                                    oc * kernel_n * kernel_m +
                                    ic * kernel_n * kernel_m * out_channels;
                                auto kernel_tile_handle =
                                    kernel.get_tile_handle(kernel_index);
                                Index kernel_tile_m_current =
                                    kernel.get_tile_traits(kernel_index)
                                        .matrix_shape[1][0];
                                Index kernel_tile_n_current =
                                    kernel.get_tile_traits(kernel_index)
                                        .matrix_shape[2][0] /
                                    kernel_tile_m_current;
                                Index kernel_offset_n =
                                    kernel_i * kernel_tile_n;
                                Index kernel_offset_m =
                                    kernel_j * kernel_tile_m;

                                for(Index dst_i = 0; dst_i < dst_n; ++dst_i)
                                {
                                    for(Index dst_j = 0; dst_j < dst_m; ++dst_j)
                                    {
                                        Index dst_index =
                                            dst_j + dst_i * dst_m +
                                            oc * dst_n * dst_m +
                                            b * out_channels * dst_n * dst_m;
                                        auto dst_tile_handle =
                                            dst.get_tile_handle(dst_index);

                                        Index dst_tile_m_current =
                                            dst.get_tile_traits(dst_index)
                                                .matrix_shape[1][0];
                                        Index dst_tile_n_current =
                                            dst.get_tile_traits(dst_index)
                                                .matrix_shape[2][0] /
                                            dst_tile_m_current;
                                        Index dst_offset_n = dst_i * dst_tile_n;
                                        Index dst_offset_m = dst_j * dst_tile_m;

                                        Index offset_n = dst_offset_n -
                                                         src_offset_n -
                                                         kernel_offset_n;
                                        Index offset_m = dst_offset_m -
                                                         src_offset_m -
                                                         kernel_offset_m;

                                        if(src_tile_n_current +
                                                   kernel_tile_n_current - 2 <
                                               offset_n ||
                                           offset_n + dst_tile_n_current - 1 <
                                               0 ||
                                           src_tile_m_current +
                                                   kernel_tile_m_current - 2 <
                                               offset_m ||
                                           offset_m + dst_tile_m_current - 1 <
                                               0)
                                            continue;
                                        {
                                            starpu::conv2d::submit<T>(
                                                offset_n, offset_m,
                                                tile_batch_current,
                                                tile_oc_current,
                                                tile_ic_current,
                                                src_tile_n_current,
                                                src_tile_m_current,
                                                src_tile_handle,
                                                kernel_tile_n_current,
                                                kernel_tile_m_current,
                                                kernel_tile_handle,
                                                dst_tile_n_current,
                                                dst_tile_m_current,
                                                dst_tile_handle);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void conv2d(const Tensor<T> &src, const Tensor<T> &kernel, const Tensor<T> &dst)
//! Tensor<T> 2D-Convolution between 2 matrices
/*! Blocking version of conv2d_async<T>.
 * Reshapes input tensors into 2-dimensional arrays
 * and performs the 2D-Convolution
 *
 * @param[in] src: Input tensor, that is reshaped into 2D array
 * @param[in] kernel: Input tensor, that is reshaped into 2D array
 * @param[out] dst: Resulting tensor, that is reshaped into 2D array
 * */
{
    conv2d_async<T>(src, kernel, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template void conv2d_async<fp32_t>(const Tensor<fp32_t> &src,
                                   const Tensor<fp32_t> &kernel,
                                   const Tensor<fp32_t> &dst);

template void conv2d_async<fp64_t>(const Tensor<fp64_t> &src,
                                   const Tensor<fp64_t> &kernel,
                                   const Tensor<fp64_t> &dst);

// Explicit instantiation of template
template void conv2d<fp32_t>(const Tensor<fp32_t> &src,
                             const Tensor<fp32_t> &kernel,
                             const Tensor<fp32_t> &dst);

template void conv2d<fp64_t>(const Tensor<fp64_t> &src,
                             const Tensor<fp64_t> &kernel,
                             const Tensor<fp64_t> &dst);

} // namespace tensor
} // namespace nntile
