/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/layer/mlp.cc
 * Multilayer perceptron
 *
 * @version 1.1.0
 * */

#include "nntile/layer/mlp.hh"
#include "nntile/tensor/distributions.hh"
#include "nntile/starpu.hh"
#include "../testing.hh"
#include <limits>
#include <cmath>

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void validate()
{
    // Wait until all previously used tags are cleaned
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Set up test layer
    TensorTraits linear1_traits({80, 20}, {6, 6}),
        linear2_traits({20, 80}, {6, 6}),
        tmp_traits({80, 30}, {6, 4});
    std::vector<int> mpi_grid = {2, 3};
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    std::vector<int> linear1_distr = distributions::block_cyclic(
            linear1_traits.grid.shape, mpi_grid, 0, mpi_size),
        linear2_distr = distributions::block_cyclic(
            linear2_traits.grid.shape, mpi_grid, 0, mpi_size),
        tmp_distr = distributions::block_cyclic(
            tmp_traits.grid.shape, mpi_grid, 0, mpi_size);
    std::vector<int> distr0 = {mpi_root};
    starpu_mpi_tag_t last_tag = 0;
    layer::MLP<T> layer(linear1_traits, linear1_distr, linear2_traits,
            linear2_distr, tmp_traits, tmp_distr, last_tag);
    unsigned long long seed0 = -1, seed1 = 1, seed2 = -100000000ULL;
    T mean = 0, stddev = 1;
    std::vector<Index> zeros(2);
    randn<T>(layer.get_linear1().get_weight(), zeros, linear1_traits.shape,
            seed0, mean, stddev);
    randn<T>(layer.get_linear2().get_weight(), zeros, linear2_traits.shape,
            seed1, mean, stddev);
    // Set up test input and output
    TensorTraits inout_traits({20, 30}, {6, 4}),
                 single_traits({20, 30}, {20, 30});
    std::vector<int> inout_distr = distributions::block_cyclic(
            inout_traits.grid.shape, mpi_grid, 0, mpi_size);
    Tensor<T> input(inout_traits, inout_distr, last_tag),
        output(inout_traits, inout_distr, last_tag),
        output2(tmp_traits, tmp_distr, last_tag),
        output3(inout_traits, inout_distr, last_tag);
    randn<T>(input, zeros, inout_traits.shape, seed2, mean, stddev);
    // Launch layer forward
    layer.forward_async(input, output);
    // Get the same in terms of tensors
    constexpr T one = 1.0, zero = 0.0;
    constexpr TransOp opN(TransOp::NoTrans);
    gemm<T>(one, opN, layer.get_linear1().get_weight(), opN, input, zero,
            output2, 1);
    gelutanh<T>(output2);
    gemm<T>(one, opN, layer.get_linear2().get_weight(), opN, output2, zero,
            output3, 1);
    // Gather results on the root node
    Tensor<T> output_single(single_traits, distr0, last_tag),
        output3_single(single_traits, distr0, last_tag);
    gather<T>(output, output_single);
    gather<T>(output3, output3_single);
    // Check results on the root node
    if(mpi_rank == mpi_root)
    {
        auto output_local = output_single.get_tile(0).acquire(STARPU_R);
        auto output3_local = output3_single.get_tile(0).acquire(STARPU_R);
        T output_max = 0, output3_max = 0, output_ssq = 1, output3_ssq = 1;
        for(Index i = 0; i < output.nelems; ++i)
        {
            T tmp1 = output_local[i], tmp3 = output3_local[i];
            tmp3 = std::abs(tmp3-tmp1);
            tmp1 = std::abs(tmp1);
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
            if(tmp3 > output3_max)
            {
                T tmp = output3_max / tmp3;
                tmp *= tmp;
                output3_max = tmp3;
                output3_ssq = output_ssq*tmp + 1;
            }
            else
            {
                T tmp = tmp3 / output3_max;
                output3_ssq += tmp * tmp;
            }
        }
        T ratio = std::sqrt(output3_ssq/output_ssq) * output3_max / output_max;
        TEST_ASSERT(ratio < 10*std::numeric_limits<T>::epsilon());
        output_local.release();
        output3_local.release();
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
