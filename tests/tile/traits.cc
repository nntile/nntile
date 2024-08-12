/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/traits.cc
 * Traits of Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/traits.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

void validate_traits(const TileTraits &traits)
{
    if(traits.shape.size() != traits.ndim)
    {
        throw std::runtime_error("Inconsistent ndim and shape.size()");
    }
    if(traits.stride.size() != traits.ndim)
    {
        throw std::runtime_error("Inconsistent ndim and stride.size()");
    }
    if(traits.matrix_shape.size() != traits.ndim+1)
    {
        throw std::runtime_error("Inconsistent ndim and matrix_shape.size()");
    }
    Index tmp = 1;
    for(Index i = 0; i < traits.ndim; ++i)
    {
        if(traits.stride[i] != tmp)
        {
            throw std::runtime_error("Inconsistent stride");
        }
        tmp *= traits.shape[i];
    }
    if(tmp != traits.nelems)
    {
        throw std::runtime_error("Inconsistent nelems");
    }
    tmp = 1;
    if(traits.matrix_shape[0][0] != 1)
    {
        throw std::runtime_error("Inconsistent matrix_shape[0][0]");
    }
    for(Index i = 1; i <= traits.ndim; ++i)
    {
        tmp *= traits.shape[i-1];
        if(traits.matrix_shape[i][0] != tmp)
        {
            throw std::runtime_error("Inconsistent matrix_shape[i][0]");
        }
    }
    for(Index i = 0; i <= traits.ndim; ++i)
    {
        tmp = traits.matrix_shape[i][0] * traits.matrix_shape[i][1];
        if(traits.nelems != tmp)
        {
            throw std::runtime_error("Inconsistent matrix_shape[i][1]");
        }
    }
    if(traits.ndim > 0)
    {
        std::vector<Index> index(traits.shape);
        for(Index i = 0; i < traits.ndim; ++i)
        {
            index[i] -= 1;
        }
        ++index[0];
        TEST_THROW(traits.index_to_linear(index));
        TEST_ASSERT(!traits.contains_index(index));
        for(Index i = 1; i < traits.ndim; ++i)
        {
            --index[i-1];
            ++index[i];
            TEST_THROW(traits.index_to_linear(index));
            TEST_ASSERT(!traits.contains_index(index));
        }
        TEST_THROW(traits.index_to_linear(
                    std::vector<Index>(traits.ndim-1, 0)));
    }
    TEST_THROW(traits.index_to_linear(std::vector<Index>(traits.ndim+1, 0)));
    TEST_THROW(traits.linear_to_index(-1));
    TEST_THROW(traits.linear_to_index(traits.nelems));
    for(Index i = 0; i < traits.nelems; ++i)
    {
        const auto index = traits.linear_to_index(i);
        TEST_ASSERT(traits.contains_index(index));
        TEST_ASSERT(i == traits.index_to_linear(index));
        std::vector<Index> index2(index);
        index2.push_back(0);
        TEST_THROW(traits.contains_index(index2));
        index2.pop_back();
        if(traits.ndim > 0)
        {
            index2.pop_back();
            TEST_THROW(traits.contains_index(index2));
        }
    }
}

int main(int argc, char **argv)
{
    // Check scalar case with empty shape
    std::vector<Index> empty;
    std::cout << "scalar case\n";
    TileTraits scalar_traits(empty);
    {volatile TileTraits x({});};
    TEST_ASSERT(scalar_traits.shape == empty);
    std::cout << scalar_traits;
    validate_traits(scalar_traits);
    TEST_ASSERT(scalar_traits.index_to_linear(empty) == 0);
    TEST_ASSERT(scalar_traits.index_to_linear({}) == 0);
    TEST_ASSERT(scalar_traits.linear_to_index(0) == std::vector<Index>());
    // Check vector case
    std::cout << "vector case\n";
    TileTraits vector_traits({10});
    TEST_ASSERT(vector_traits.shape == std::vector<Index>{10});
    TEST_THROW(TileTraits({0}));
    std::cout << vector_traits;
    validate_traits(vector_traits);
    // Check matrix case
    std::cout << "matrix case\n";
    TileTraits matrix_traits({3, 5});
    TEST_ASSERT(matrix_traits.shape == std::vector<Index>{3, 5});
    TEST_THROW(TileTraits({3, 0}));
    TEST_THROW(TileTraits({0, 5}));
    std::cout << matrix_traits;
    validate_traits(matrix_traits);
    // Check 5-dimensional tensor
    std::cout << "5D-tensor case\n";
    TileTraits t5d_traits({1, 2, 3, 4, 5});
    TEST_THROW(TileTraits({1, 2, 3, 4, 0}));
    TEST_THROW(TileTraits({1, 2, 3, 0, 5}));
    TEST_THROW(TileTraits({1, 2, 0, 4, 5}));
    TEST_THROW(TileTraits({1, 0, 3, 4, 5}));
    TEST_THROW(TileTraits({0, 2, 3, 4, 5}));
    TEST_ASSERT(t5d_traits.shape == std::vector<Index>({1, 2, 3, 4, 5}));
    std::cout << t5d_traits;
    validate_traits(t5d_traits);
    return 0;
}
