/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/layer/linear.cc
 * Fully connected dense linear layer
 *
 * @version 1.1.0
 * */

#include "nntile/layer/linear.hh"
#include "nntile/tensor.hh"
#include "nntile/starpu.hh"
#include "../testing.hh"
#include <limits>
#include <cmath>

using namespace nntile;
using namespace nntile::tensor;
using namespace nntile::layer;

template<typename T>
void validate()
{
    // Wait until all previously used tags are cleaned
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Set up test layer
    TensorTraits input_traits({30, 20}, {5, 4}),
        output_traits({10, 20, 20}, {3, 4, 4}),
        weight_traits({10, 20, 30}, {3, 4, 5});
    std::vector<int> mpi_grid3 = {3, 2, 2}, mpi_grid2 = {2, 3};
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    std::vector<int> weight_distr = distributions::block_cyclic(
            weight_traits.grid.shape, mpi_grid3, 0, mpi_size);
    std::vector<int> distr0 = {mpi_root};
    starpu_mpi_tag_t last_tag = 0;
    Tensor<T> weight(weight_traits, weight_distr, last_tag),
        grad_weight(weight_traits, weight_distr, last_tag);
    Linear<T> layer(input_traits, output_traits, weight, grad_weight);
    unsigned long long seed0 = -1, seed1 = 1, seed2 = -100000000ULL;
    T mean = 0, stddev = 1;
    std::vector<Index> zeros2(2), zeros3(3);
    randn<T>(layer.weight, zeros3, weight_traits.shape, seed0, mean,
            stddev);
    // Set up test input and output
    TensorTraits output_single_traits({10, 20, 20}, {10, 20, 20});
    std::vector<int> input_distr = distributions::block_cyclic(
            input_traits.grid.shape, mpi_grid2, 0, mpi_size),
        output_distr = distributions::block_cyclic(
            output_traits.grid.shape, mpi_grid3, 0, mpi_size);
    Tensor<T> input(input_traits, input_distr, last_tag),
        output(output_traits, output_distr, last_tag),
        output2(output_traits, output_distr, last_tag);
    randn<T>(input, zeros2, input_traits.shape, seed1, mean, stddev);
    randn<T>(output, zeros3, output_traits.shape, seed2, mean, stddev);
    copy<T>(output, output2);
    // Launch layer forward
    layer.forward_async(input, output);
    // Get the same in terms of tensors
    constexpr T one = 1.0, zero = 0.0;
    constexpr TransOp opN(TransOp::NoTrans);
    gemm<T>(one, opN, layer.weight, opN, input, zero, output2, 1);
    // Gather results on the root node
    Tensor<T> output_single(output_single_traits, distr0, last_tag),
        output2_single(output_single_traits, distr0, last_tag);
    gather<T>(output, output_single);
    gather<T>(output2, output2_single);
    // Check results on the root node
    if(mpi_rank == mpi_root)
    {
        auto output_local = output_single.get_tile(0).acquire(STARPU_R);
        auto output2_local = output2_single.get_tile(0).acquire(STARPU_R);
        T output_max = 0, output2_max = 0, output_ssq = 1, output2_ssq = 1;
        for(Index i = 0; i < output.nelems; ++i)
        {
            T tmp1 = output_local[i], tmp2 = output2_local[i];
            tmp2 = std::abs(tmp2-tmp1);
            tmp1 = std::abs(tmp1);
            if(tmp1 > 0)
            {
                if(tmp1 > output_max)
                {
                    T tmp = output_max / tmp1;
                    tmp *= tmp;
                    output_max = tmp1;
                    output_ssq = output_ssq*tmp + 1;
                }
                else
                {
                    T tmp = tmp1 / output_max;
                    output_ssq += tmp * tmp;
                }
            }
            if(tmp2 > 0)
            {
                if(tmp2 > output2_max)
                {
                    T tmp = output2_max / tmp2;
                    tmp *= tmp;
                    output2_max = tmp2;
                    output2_ssq = output_ssq*tmp + 1;
                }
                else
                {
                    T tmp = tmp2 / output2_max;
                    output2_ssq += tmp * tmp;
                }
            }
        }
        T ratio = std::sqrt(output2_ssq/output_ssq) * output2_max / output_max;
        TEST_ASSERT(ratio < 10*std::numeric_limits<T>::epsilon());
        output_local.release();
        output2_local.release();
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
