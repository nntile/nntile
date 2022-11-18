/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/model/deep_linear.cc
 * Deep linear meural network
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-17
 * */

#include "nntile/model/deep_linear.hh"
#include "nntile/starpu.hh"
#include "../testing.hh"
#include <limits>
#include <cmath>

using namespace nntile;
using namespace nntile::tensor;
using namespace nntile::model;

template<typename T>
void validate()
{
    // Wait until all previously used tags are cleaned
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Set up test model
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init all codelets
    starpu::init();
    // Launch tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

