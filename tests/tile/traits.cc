#include "nntile/tile/traits.hh"
#include "../testing.hh"

using namespace nntile;

void validate_traits(const TileTraits &traits)
{
    if(traits.shape.size() != traits.ndim)
    {
        throw std::runtime_error("Inconsistent ndim and shape.size()");
    }
    for(size_t i = 0; i < traits.ndim; ++i)
    {
        if(traits.shape[i] < 0)
        {
            throw std::runtime_error("Negative shape");
        }
    }
    size_t tmp = 1;
    for(size_t i = 0; i < traits.ndim; ++i)
    {
        if(traits.stride[i] != tmp)
        {
            throw std::runtime_error("Inconsistent stride");
        }
        tmp *= traits.shape[i];
    }
    tmp = 1;
    if(traits.matrix_shape[0][0] != 1)
    {
        throw std::runtime_error("Inconsistent matrix_shape[0][0]");
    }
    for(size_t i = 1; i <= traits.ndim; ++i)
    {
        tmp *= traits.shape[i-1];
        if(traits.matrix_shape[i][0] != tmp)
        {
            throw std::runtime_error("Inconsistent matrix_shape[i][0]");
        }
    }
    for(size_t i = 0; i <= traits.ndim; ++i)
    {
        tmp = traits.matrix_shape[i][0] * traits.matrix_shape[i][1];
        if(traits.nelems != tmp)
        {
            throw std::runtime_error("Inconsistent matrix_shape[i][1]");
        }
    }
}

int main(int argc, char **argv)
{
    // Check scalar case with empty shape
    TileTraits scalar_traits({});
    std::cout << "scalar case\n" << scalar_traits;
    validate_traits(scalar_traits);
    TESTA((scalar_traits.get_offset({}) == 0));
    TESTN((scalar_traits.get_offset({1})));
    // Check vector case
    TileTraits vector_traits({10});
    std::cout << "vector case\n" << vector_traits;
    validate_traits(vector_traits);
    TESTA((vector_traits.get_offset({5}) == 5));
    TESTN((vector_traits.get_offset({10})));
    TESTN((vector_traits.get_offset({0, 0})));
    TESTN((vector_traits.get_offset({})));
    // Check matrix case
    TileTraits matrix_traits({3, 5});
    std::cout << "matrix case\n" << matrix_traits;
    validate_traits(matrix_traits);
    TESTA((matrix_traits.get_offset({1, 1}) == 4));
    TESTN((matrix_traits.get_offset({3, 2})));
    TESTN((matrix_traits.get_offset({2, 5})));
    TESTN((matrix_traits.get_offset({1})));
    TESTN((matrix_traits.get_offset({1, 1, 1})));
    // Check 5-dimensional tensor
    TileTraits t5d_traits({7, 9, 11, 13, 15});
    std::cout << "5D-tensor case\n" << t5d_traits;
    validate_traits(t5d_traits);
    TESTA((t5d_traits.get_offset({0, 1, 2, 3, 4}) == 38248));
    TESTN((t5d_traits.get_offset({0, 1, 2, 3, 15})));
    TESTN((t5d_traits.get_offset({0, 1, 2, 3})));
    TESTN((t5d_traits.get_offset({0, 1, 2, 3, 4, 5})));
    // Check constructor from vector
    TileTraits traits(std::vector<size_t>{1, 2, 3, 4});
    validate_traits(traits);
    return 0;
}

