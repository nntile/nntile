#include "nntile/tensor/traits.hh"
#include "../testing.hh"

using namespace nntile;

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
        TESTA(traits.grid.index_to_linear(grid_index) == i);
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
    TESTA(scalar_traits.shape == std::vector<Index>());
    TESTA(scalar_traits.basetile_shape == std::vector<Index>());
    TESTA(scalar_traits.leftover_shape == std::vector<Index>());
    TESTA(scalar_traits.grid.shape == std::vector<Index>());
    TESTA(scalar_traits.get_tile_shape({}) == empty);
    TESTN(scalar_traits.get_tile_shape({1}));
    TESTP(TensorTraits({}, {}));
    TESTN(TensorTraits({}, {1}));
    TESTN(TensorTraits({1}, {}));
    // Check vector case
    std::cout << "vector case\n";
    TensorTraits vector_traits({10}, {3});
    std::cout << vector_traits;
    validate_traits(vector_traits);
    TESTA(vector_traits.shape == std::vector<Index>{10});
    TESTA(vector_traits.basetile_shape == std::vector<Index>{3});
    TESTA(vector_traits.leftover_shape == std::vector<Index>{1});
    TESTA(vector_traits.grid.shape == std::vector<Index>{4});
    TESTN(vector_traits.get_tile_shape({-1}));
    TESTN(vector_traits.get_tile_shape({4}));
    TESTA(vector_traits.get_tile_shape({3}) == std::vector<Index>{1});
    TESTA(vector_traits.get_tile_shape({2}) == std::vector<Index>{3});
    TESTN(TensorTraits({0}, {1}));
    TESTN(TensorTraits({1}, {0}));
    TESTN(TensorTraits({1}, {1, 1}));
    TESTN(TensorTraits({1, 1}, {1}));
    // Check matrix case
    std::cout << "matrix case\n";
    TensorTraits matrix_traits({3, 5}, {3, 5});
    validate_traits(matrix_traits);
    std::cout << matrix_traits;
    TESTN(matrix_traits.get_tile_shape({0, -1}));
    TESTN(matrix_traits.get_tile_shape({-1, 0}));
    TESTN(matrix_traits.get_tile_shape({0, 1}));
    TESTN(matrix_traits.get_tile_shape({1, 0}));
    TESTA((matrix_traits.get_tile_shape({0, 0}) == std::vector<Index>{3, 5}));
    TESTN(TensorTraits({3, 5}, {3, 0}));
    TESTN(TensorTraits({3, 5}, {0, 5}));
    TESTN(TensorTraits({3, 0}, {3, 5}));
    TESTN(TensorTraits({0, 5}, {3, 5}));
    // Check 5-dimensional tensor
    std::cout << "5D-tensor case\n";
    TensorTraits t5d_traits({7, 9, 11, 13, 15}, {100, 100, 100, 100, 100});
    std::cout << t5d_traits;
    validate_traits(t5d_traits);
    TESTA(t5d_traits.shape == std::vector<Index>({7, 9, 11, 13, 15}));
    TESTA(t5d_traits.basetile_shape ==
            std::vector<Index>({100, 100, 100, 100, 100}));
    TESTA(t5d_traits.leftover_shape == std::vector<Index>({7, 9, 11, 13, 15}));
    return 0;
}

