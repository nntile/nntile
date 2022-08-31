/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/distributions.cc
 * Distributions for tesnors
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-29
 * */

#include "nntile/tensor/distributions.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;
using namespace nntile::tensor::distributions;

void validate()
{
    std::vector<Index> tensor_grid({5});
    std::vector<int> mpi_grid({3});
    int start_rank = 1;
    int max_rank = 2;
    std::vector<int> distr({1, 0, 1, 1, 0});
    TEST_ASSERT(distr == block_cyclic(tensor_grid, mpi_grid, start_rank,
                max_rank));
    std::vector<Index> tensor_grid2({5, 3});
    std::vector<int> mpi_grid2({3, 2});
    int start_rank2 = 1;
    int max_rank2 = 6;
    std::vector<int> distr2({1, 2, 3, 1, 2, 4, 5, 0, 4, 5, 1, 2, 3, 1, 2});
    TEST_ASSERT(distr2 == block_cyclic(tensor_grid2, mpi_grid2, start_rank2,
                max_rank2));
    std::vector<Index> tensor_grid3({3, 3, 2});
    std::vector<int> mpi_grid3({1, 4, 2});
    int start_rank3 = 1;
    int max_rank3 = 8;
    std::vector<int> distr3({1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 5, 5, 6, 6, 6, 7,
            7, 7});
    TEST_ASSERT(distr3 == block_cyclic(tensor_grid3, mpi_grid3, start_rank3,
                max_rank3));
}

int main(int argc, char ** argv)
{
    validate();
    return 0;
}

