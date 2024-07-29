/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/conv2d.cc
 * 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/conv2d.hh"
#include "../testing.hh"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace nntile;
using namespace nntile::kernel::conv2d;

// Templated validation
template <typename T>
void validate(Index offset_x, Index offset_y, Index batch, Index out_channels,
              Index in_channels, Index padding_n, Index limit_n,
              Index padding_m, Index limit_m, Index nx, Index ny, Index mx,
              Index my, Index kx, Index ky, Index px, Index py, Index qx,
              Index qy)
{
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // Init test input
    std::vector<T> src(nx * ny * in_channels * batch),
        kernel(mx * my * in_channels * out_channels),
        dst(kx * ky * out_channels * batch);

    for(Index i = 0; i < kx * ky * out_channels * batch; ++i)
    {
        dst[i] = -2;
    }

    for(Index i = 0; i < nx * ny * in_channels * batch; ++i)
    {
        src[i] = 0;
    }
    for(Index i = 0; i < (batch < in_channels ? batch : in_channels); ++i)
    {
        src[px * ny + py + nx * ny * i + nx * ny * in_channels * i] = 1;
    }

    for(Index i = 0; i < mx * my * in_channels * out_channels; ++i)
    {
        kernel[i] = 0;
    }
    for(Index i = 0;
        i < (out_channels < in_channels ? out_channels : in_channels); ++i)
    {
        kernel[qx * my + qy + mx * my * i + mx * my * out_channels * i] = 1;
    }

    // Save original dst
    std::vector<T> dst_save(dst);
    // Check low-level CPU kernel
    std::cout << "Run kernel::conv2d::cpu<T>\n";
    cpu<T>(offset_x, offset_y, batch, out_channels, in_channels, padding_n,
           limit_n, padding_m, limit_m, nx, ny, &src[0], mx, my, &kernel[0], kx,
           ky, &dst[0]);
    for(Index b = 0; b < batch; ++b)
    {
        for(Index c = 0; c < out_channels; ++c)
        {
            for(Index i0 = 0; i0 < ky; ++i0)
            {
                for(Index i1 = 0; i1 < kx; ++i1)
                {
                    T ref = -1;
                    if(i0 != py + qy - offset_y - padding_m)
                        ref = -2;
                    if(i1 != px + qx - offset_x - padding_n)
                        ref = -2;
                    if(px < padding_n || py < padding_m)
                        ref = -2;
                    if(px >= limit_n)
                        ref = -2;
                    if(py >= limit_m)
                        ref = -2;
                    if(b != c)
                        ref = -2;

                    T val = dst[i1 * ky + i0 + ky * kx * c +
                                ky * kx * out_channels * b];
                    TEST_ASSERT(std::abs(val - ref) <= 10 * eps);
                }
            }
        }
    }
    std::cout << "OK: kernel::conv2d::cpu<T>\n";
}

void validate_all(Index offset_x, Index offset_y, Index batch,
                  Index out_channels, Index in_channels, Index padding_n,
                  Index limit_n, Index padding_m, Index limit_m, Index nx,
                  Index ny, Index mx, Index my, Index kx, Index ky, Index px,
                  Index py, Index qx, Index qy)
{
    validate<fp32_t>(offset_x, offset_y, batch, out_channels, in_channels,
                     padding_n, limit_n, padding_m, limit_m, nx, ny, mx, my, kx,
                     ky, px, py, qx, qy);
    validate<fp64_t>(offset_x, offset_y, batch, out_channels, in_channels,
                     padding_n, limit_n, padding_m, limit_m, nx, ny, mx, my, kx,
                     ky, px, py, qx, qy);
}

int main(int argc, char **argv)
{
    // 4x4 by 1x1, no extra arguments
    validate_all(0, 0, 1, 1, 1, 0, 4, 0, 4, 4, 4, 1, 1, 4, 4, 0, 0, 0, 0);
    // 4x4 by 6x6
    validate_all(0, 0, 1, 1, 1, 0, 4, 0, 4, 4, 4, 6, 6, 9, 9, 0, 0, 0, 0);
    // 5x3 by 7x11
    validate_all(0, 0, 1, 1, 1, 0, 5, 0, 3, 5, 3, 7, 11, 11, 13, 0, 0, 0, 0);
    // Random pixel in the middle instead of (0,0)
    validate_all(0, 0, 1, 1, 1, 0, 5, 0, 3, 5, 3, 7, 11, 11, 13, 3, 1, 5, 6);
    // Different offsets
    validate_all(1, 0, 1, 1, 1, 0, 5, 0, 3, 5, 3, 7, 11, 11, 13, 3, 1, 5, 6);
    validate_all(0, 1, 1, 1, 1, 0, 5, 0, 3, 5, 3, 7, 11, 11, 13, 3, 1, 5, 6);
    validate_all(-1, 0, 1, 1, 1, 0, 5, 0, 3, 5, 3, 7, 11, 11, 13, 3, 1, 5, 6);
    validate_all(0, -1, 1, 1, 1, 0, 5, 0, 3, 5, 3, 7, 11, 11, 13, 3, 1, 5, 6);
    // Padding on the beginning of the axes
    validate_all(0, 0, 1, 1, 1, 1, 5, 0, 3, 5, 3, 7, 11, 11, 13, 3, 1, 5, 6);
    validate_all(0, 0, 1, 1, 1, 0, 5, 1, 3, 5, 3, 7, 11, 11, 13, 3, 1, 5, 6);
    // Element outside of padding
    validate_all(0, 0, 1, 1, 1, 1, 5, 0, 3, 5, 3, 7, 11, 11, 13, 0, 1, 5, 6);
    validate_all(0, 0, 1, 1, 1, 0, 5, 1, 3, 5, 3, 7, 11, 11, 13, 3, 1, 0, 6);
    // Padding on the end of the axes
    validate_all(0, 0, 1, 1, 1, 0, 1, 0, 3, 5, 3, 7, 11, 11, 13, 3, 1, 5, 6);
    validate_all(0, 0, 1, 1, 1, 0, 5, 0, 1, 5, 3, 7, 11, 11, 13, 3, 1, 5, 6);
    // Element outside of padding
    validate_all(0, 0, 1, 1, 1, 0, 1, 0, 3, 5, 3, 7, 11, 11, 13, 0, 1, 5, 6);
    validate_all(0, 0, 1, 1, 1, 0, 5, 0, 1, 5, 3, 7, 11, 11, 13, 3, 1, 0, 6);
    // In_channels, out_channels, batch not zero in different combinations
    validate_all(0, 0, 1, 1, 6, 0, 5, 0, 3, 5, 3, 7, 11, 11, 13, 3, 1, 5, 6);
    validate_all(0, 0, 1, 4, 1, 0, 5, 0, 3, 5, 3, 7, 11, 11, 13, 3, 1, 5, 6);
    validate_all(0, 0, 8, 1, 1, 0, 5, 0, 3, 5, 3, 7, 11, 11, 13, 3, 1, 5, 6);
    validate_all(0, 0, 8, 4, 6, 0, 5, 0, 3, 5, 3, 7, 11, 11, 13, 3, 1, 5, 6);
    // Some random combination of all the modifiers with bigger sizes
    validate_all(17, 13, 8, 4, 6, 11, 48, 21, 56, 51, 61, 70, 40, 110, 100, 37,
                 33, 10, 12);

    return 0;
}
