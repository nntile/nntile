/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/layer/gelutanh.cc
 * Approximate GeLU layer
 *
 * @version 1.1.0
 * */

#include "nntile/layer/gelutanh.hh"
#include "nntile/tensor/distributions.hh"
#include "nntile/starpu.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void validate()
{
    // Wait until all previously used tags are cleaned
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Set up test layer
    layer::GeLUTanh<T> layer;
    // Set up test input
    std::vector<int> mpi_grid = {2, 2};
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    starpu_mpi_tag_t last_tag = 0;
    TensorTraits traits({30, 20}, {5, 4}),
                 single_traits({30, 20}, {30, 20});
    std::vector<int> distr = distributions::block_cyclic(
            traits.grid.shape, mpi_grid, 0, mpi_size),
        distr0 = {mpi_root};
    Tensor<T> Ain(traits, distr, last_tag), Aout(traits, distr, last_tag),
        B(single_traits, distr0, last_tag), C(single_traits, distr0, last_tag);
    unsigned long long seed = -1;
    T mean = 0, stddev = 1;
    std::vector<Index> zeros = {0, 0};
    randn<T>(Ain, zeros, traits.shape, seed, mean, stddev);
    gather<T>(Ain, B);
    // Launch layer forward
    layer.forward_async(Ain, Aout);
    // Gather results on the root node
    gather<T>(Aout, C);
    // Get reference result on the root node
    gelutanh<T>(B);
    // Check results on the root node
    if(mpi_rank == mpi_root)
    {
        auto B_local = B.get_tile(0).acquire(STARPU_R);
        auto C_local = C.get_tile(0).acquire(STARPU_R);
        for(Index i = 0; i < B.nelems; ++i)
        {
            TEST_ASSERT(B_local[i] == C_local[i]);
        }
        B_local.release();
        C_local.release();
    }
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
