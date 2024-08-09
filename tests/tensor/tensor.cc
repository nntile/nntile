/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/tensor.cc
 * Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/tensor.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(const Tensor<T> &A)
{
    TEST_THROW(A.get_tile(-1));
    TEST_THROW(A.get_tile(A.grid.nelems));
    auto index = A.grid.shape;
    for(Index i = 0; i < A.ndim; ++i)
    {
        --index[i];
    }
    if(A.ndim > 0)
    {
        ++index[0];
        TEST_THROW(A.get_tile(index));
        for(Index i = 1; i < A.ndim; ++i)
        {
            --index[i-1];
            ++index[i];
            TEST_THROW(A.get_tile(index));
        }
        TEST_THROW(A.get_tile(std::vector<Index>(A.ndim-1)));
    }
    TEST_THROW(A.get_tile(std::vector<Index>(A.ndim+1)));
    for(Index i = 0; i < A.grid.nelems; ++i)
    {
        const auto tile_index = A.grid.linear_to_index(i);
        TEST_ASSERT(A.get_tile_shape(tile_index)
                == A.get_tile(tile_index).shape);
    }
}

template<typename T>
void validate()
{
    starpu_mpi_tag_t last_tag = 0;
    TensorTraits scalar_traits({}, {});
    std::vector<int> scalar_distr = {1};
    Tensor<T> scalar(scalar_traits, scalar_distr, last_tag);
    TEST_ASSERT(scalar.get_tile(0).mpi_get_rank() == 1);
    check<T>(scalar);
    TensorTraits vector_traits({10}, {3});
    std::vector<int> vector_distr = {1, 3, 7, 2};
    Tensor<T> vector(vector_traits, vector_distr, last_tag);
    for(Index i = 0; i < vector_distr.size(); ++i)
    {
        TEST_ASSERT(vector.get_tile(i).mpi_get_rank() == vector_distr[i]);
    }
    check<T>(vector);
    TensorTraits matrix_traits({3, 5}, {3, 5});
    std::vector<int> matrix_distr = {3};
    Tensor<T> matrix(matrix_traits, matrix_distr, last_tag);
    TEST_ASSERT(matrix.get_tile(0).mpi_get_rank() == 3);
    check<T>(matrix);
    TensorTraits t5d_traits({11, 13, 15, 17, 19}, {100, 100, 100, 100, 100});
    std::vector<int> t5d_distr = {4};
    Tensor<T> t5d(t5d_traits, t5d_distr, last_tag);
    check<T>(t5d);
    TensorTraits t5d2_traits({40, 40, 40, 40, 40}, {11, 13, 15, 17, 19});
    std::vector<int> t5d2_distr(4*4*3*3*3);
    for(Index i = 0; i < t5d2_distr.size(); ++i)
    {
        t5d2_distr[i] = i+3;
    }
    Tensor<T> t5d2(t5d2_traits, t5d2_distr, last_tag);
    for(Index i = 0; i < t5d2_distr.size(); ++i)
    {
        TEST_ASSERT(t5d2.get_tile(i).mpi_get_rank() == i+3);
    }
    check<T>(t5d2);
}

int main(int argc, char ** argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
