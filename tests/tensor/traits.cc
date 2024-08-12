/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/traits.cc
 * Traits of Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/traits.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;
using namespace nntile::tensor;

void validate_traits(const TensorTraits &traits)
{
    for(Index i = 0; i < traits.ndim; ++i)
    {
        if(traits.grid.shape[i] == 0 and traits.shape[i] != 0)
        {
            throw std::runtime_error("grid.shape[i] == 0 and shape[i] != 0");
        }
        else if(traits.grid.shape[i] != 0)
        {
            Index tmp = traits.grid.shape[i] - 1;
            tmp = tmp*traits.basetile_shape[i] + traits.leftover_shape[i];
            if(traits.shape[i] != tmp)
            {
                throw std::runtime_error("traits.shape[i] != tmp");
            }
        }
    }
    Index nelems = 0;
    for(Index i = 0; i < traits.grid.nelems; ++i)
    {
        auto grid_index = traits.grid.linear_to_index(i);
        TEST_ASSERT(traits.grid.index_to_linear(grid_index) == i);
        nelems += TileTraits(traits.get_tile_shape(grid_index)).nelems;
    }
    if(traits.nelems != nelems)
    {
        throw std::runtime_error("traits.nelem != nelem");
    }
}


int main(int argc, char **argv)
{
    std::vector<Index> empty;
    // Check scalar case with empty shape
    std::cout << "scalar case\n";
    TensorTraits scalar_traits(empty, empty);
    std::cout << scalar_traits;
    validate_traits(scalar_traits);
    TEST_ASSERT(scalar_traits.shape == std::vector<Index>());
    TEST_ASSERT(scalar_traits.basetile_shape == std::vector<Index>());
    TEST_ASSERT(scalar_traits.leftover_shape == std::vector<Index>());
    TEST_ASSERT(scalar_traits.grid.shape == std::vector<Index>());
    TEST_ASSERT(scalar_traits.get_tile_shape({}) == empty);
    TEST_THROW(scalar_traits.get_tile_shape({1}));
    {volatile TensorTraits x({}, {});};
    TEST_THROW(TensorTraits({}, {1}));
    TEST_THROW(TensorTraits({1}, {}));
    // Check vector case
    std::cout << "vector case\n";
    TensorTraits vector_traits({10}, {3});
    std::cout << vector_traits;
    validate_traits(vector_traits);
    TEST_ASSERT(vector_traits.shape == std::vector<Index>{10});
    TEST_ASSERT(vector_traits.basetile_shape == std::vector<Index>{3});
    TEST_ASSERT(vector_traits.leftover_shape == std::vector<Index>{1});
    TEST_ASSERT(vector_traits.grid.shape == std::vector<Index>{4});
    TEST_THROW(vector_traits.get_tile_shape({-1}));
    TEST_THROW(vector_traits.get_tile_shape({4}));
    TEST_ASSERT(vector_traits.get_tile_shape({3}) == std::vector<Index>{1});
    TEST_ASSERT(vector_traits.get_tile_shape({2}) == std::vector<Index>{3});
    TEST_THROW(TensorTraits({0}, {1}));
    TEST_THROW(TensorTraits({1}, {0}));
    TEST_THROW(TensorTraits({1}, {1, 1}));
    TEST_THROW(TensorTraits({1, 1}, {1}));
    // Check matrix case
    std::cout << "matrix case\n";
    TensorTraits matrix_traits({3, 5}, {3, 5});
    validate_traits(matrix_traits);
    std::cout << matrix_traits;
    TEST_THROW(matrix_traits.get_tile_shape({0, -1}));
    TEST_THROW(matrix_traits.get_tile_shape({-1, 0}));
    TEST_THROW(matrix_traits.get_tile_shape({0, 1}));
    TEST_THROW(matrix_traits.get_tile_shape({1, 0}));
    TEST_ASSERT((matrix_traits.get_tile_shape({0, 0}) == std::vector<Index>{3, 5}));
    TEST_THROW(TensorTraits({3, 5}, {3, 0}));
    TEST_THROW(TensorTraits({3, 5}, {0, 5}));
    TEST_THROW(TensorTraits({3, 0}, {3, 5}));
    TEST_THROW(TensorTraits({0, 5}, {3, 5}));
    // Check 5-dimensional tensor
    std::cout << "5D-tensor case\n";
    TensorTraits t5d_traits({7, 9, 11, 13, 15}, {100, 100, 100, 100, 100});
    std::cout << t5d_traits;
    validate_traits(t5d_traits);
    TEST_ASSERT(t5d_traits.shape == std::vector<Index>({7, 9, 11, 13, 15}));
    TEST_ASSERT(t5d_traits.basetile_shape ==
            std::vector<Index>({100, 100, 100, 100, 100}));
    TEST_ASSERT(t5d_traits.leftover_shape == std::vector<Index>({7, 9, 11, 13, 15}));
    return 0;
}
