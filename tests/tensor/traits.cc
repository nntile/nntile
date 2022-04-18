#include "nntile/tensor/traits.hh"
#include "../testing.hh"

using namespace nntile;

void validate_traits(const TensorTraits &traits)
{
    for(size_t i = 0; i < traits.ndim; ++i)
    {
        if(traits.grid.shape[i] == 0 and traits.shape[i] != 0)
        {
            throw std::runtime_error("grid.shape[i] == 0 and shape[i] != 0");
        }
        else if(traits.grid.shape[i] != 0)
        {
            size_t tmp = traits.grid.shape[i] - 1;
            tmp = tmp*traits.basetile_shape[i] + traits.leftover_shape[i];
            if(traits.shape[i] != tmp)
            {
                throw std::runtime_error("traits.shape[i] != tmp");
            }
        }
    }
    for(size_t i = 0; i < traits.grid.nelems; ++i)
    {
        TESTA(traits.get_tile_offset(traits.get_tile_index(i)) == i);
    }
}


int main(int argc, char **argv)
{
    // Check scalar case with empty shape
    TensorTraits scalar_traits({}, {});
    TESTA(scalar_traits.shape == std::vector<size_t>());
    TESTA(scalar_traits.basetile_shape == std::vector<size_t>());
    TESTA(scalar_traits.leftover_shape == std::vector<size_t>());
    TESTA(scalar_traits.grid.shape == std::vector<size_t>());
    scalar_traits.get_tile_index(0);
    scalar_traits.get_tile_offset({});
    scalar_traits.get_tile_shape({});
    TESTN(scalar_traits.get_tile_index(1));
    TESTN(scalar_traits.get_tile_shape({1}));
    std::cout << "scalar case\n" << scalar_traits;
    validate_traits(scalar_traits);
    // Check vector case
    TensorTraits vector_traits({10}, {2});
    std::cout << "vector case\n" << vector_traits;
    validate_traits(vector_traits);
    vector_traits.get_tile_offset({4});
    TESTN(vector_traits.get_tile_shape(std::vector<size_t>{5}));
    vector_traits.get_tile_shape(4);
    TESTN((vector_traits.get_tile_offset({5})));
    TESTN((vector_traits.get_tile_offset({0, 0})));
    TESTN((vector_traits.get_tile_offset({})));
    // Check matrix case
    TensorTraits matrix_traits({3, 5}, {3, 5});
    std::cout << "matrix case\n" << matrix_traits;
    validate_traits(matrix_traits);
    matrix_traits.get_tile_offset({0, 0});
    TESTN((matrix_traits.get_tile_offset({3, 2})));
    TESTN((matrix_traits.get_tile_offset({2, 5})));
    TESTN((matrix_traits.get_tile_offset({1})));
    TESTN((matrix_traits.get_tile_offset({1, 1, 1})));
    // Check 5-dimensional tensor
    TensorTraits t5d_traits({7, 9, 11, 13, 15}, {100, 100, 100, 100, 100});
    TESTA(t5d_traits.shape == std::vector<size_t>({7, 9, 11, 13, 15}));
    std::cout << "5D-tensor case\n" << t5d_traits;
    validate_traits(t5d_traits);
    t5d_traits.get_tile_offset({0, 0, 0, 0, 0});
    TESTN((t5d_traits.get_tile_offset({0, 1, 2, 3, 15})));
    TESTN((t5d_traits.get_tile_offset({0, 1, 2, 3})));
    TESTN((t5d_traits.get_tile_offset({0, 1, 2, 3, 4, 5})));
    // Other checks
    TESTN(TensorTraits({1}, {1, 2}));
    validate_traits(TensorTraits({0}, {0}));
    TESTN(TensorTraits({0}, {0}).get_tile_offset({0}));
    validate_traits(TensorTraits({1, 2, 0}, {4, 1, 0}));
    TESTN(TensorTraits({1, 2, 0}, {4, 0, 0}));
    validate_traits(TensorTraits({1, 0, 0}, {4, 1, 0}));
    return 0;
}

