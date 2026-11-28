/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file tests/kernel/conv2d_bwd_weight_inplace.cc
 * Placeholder for conv2d_bwd_weight_inplace kernel test
 *
 * @version 2.0.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#include "nntile/kernel/conv2d_bwd_weight_inplace.hh"
#include "../testing.hh"
#include <iostream>

using namespace nntile;
using namespace nntile::kernel;

int main(int argc, char **argv)
{
    // This is a placeholder test.
    std::cout << "STUB: This is a placeholder for the conv2d_bwd_weight_inplace kernel test." << std::endl;
    return 0;
}
