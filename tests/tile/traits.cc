#include "nntile/tile/traits.hh"
#include "../testing.hh"

using namespace nntile;

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
        TESTN(traits.index_to_linear(index));
        TESTA(!traits.contains_index(index));
        for(Index i = 1; i < traits.ndim; ++i)
        {
            --index[i-1];
            ++index[i];
            TESTN(traits.index_to_linear(index));
            TESTA(!traits.contains_index(index));
        }
        TESTN(traits.index_to_linear(std::vector<Index>(traits.ndim-1, 0)));
    }
    TESTN(traits.index_to_linear(std::vector<Index>(traits.ndim+1, 0)));
    TESTN(traits.linear_to_index(-1));
    TESTN(traits.linear_to_index(traits.nelems));
    for(Index i = 0; i < traits.nelems; ++i)
    {
        const auto index = traits.linear_to_index(i);
        TESTA(traits.contains_index(index));
        TESTA(i == traits.index_to_linear(index));
    }
}

int main(int argc, char **argv)
{
    // Check scalar case with empty shape
    std::vector<Index> empty;
    std::cout << "scalar case\n";
    TileTraits scalar_traits(empty);
    TESTP(TileTraits({}));
    TESTA(scalar_traits.shape == empty);
    std::cout << scalar_traits;
    validate_traits(scalar_traits);
    TESTA(scalar_traits.index_to_linear(empty) == 0);
    TESTA(scalar_traits.index_to_linear({}) == 0);
    TESTA(scalar_traits.linear_to_index(0) == std::vector<Index>());
    // Check vector case
    std::cout << "vector case\n";
    TileTraits vector_traits({10});
    TESTA(vector_traits.shape == std::vector<Index>{10});
    TESTN(TileTraits({0}));
    std::cout << vector_traits;
    validate_traits(vector_traits);
    // Check matrix case
    std::cout << "matrix case\n";
    TileTraits matrix_traits({3, 5});
    TESTA((matrix_traits.shape == std::vector<Index>{3, 5}));
    TESTN(TileTraits({3, 0}));
    TESTN(TileTraits({0, 5}));
    std::cout << matrix_traits;
    validate_traits(matrix_traits);
    // Check 5-dimensional tensor
    std::cout << "5D-tensor case\n";
    TileTraits t5d_traits({7, 9, 11, 13, 15});
    TESTN(TileTraits({7, 9, 11, 13, 0}));
    TESTN(TileTraits({7, 9, 11, 0, 15}));
    TESTN(TileTraits({7, 9, 0, 13, 15}));
    TESTN(TileTraits({7, 0, 11, 13, 15}));
    TESTN(TileTraits({0, 9, 11, 13, 15}));
    TESTA(t5d_traits.shape == std::vector<Index>({7, 9, 11, 13, 15}));
    std::cout << t5d_traits;
    validate_traits(t5d_traits);
    return 0;
}

