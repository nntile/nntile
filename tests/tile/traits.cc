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
    if(traits.offset.size() != traits.ndim)
    {
        throw std::runtime_error("Inconsistent ndim and offset.size()");
    }
    if(traits.underlying_shape.size() != traits.ndim)
    {
        throw std::runtime_error("Inconsistent ndim and "
                "underlying_shape.size()");
    }
    if(traits.underlying_stride.size() != traits.ndim)
    {
        throw std::runtime_error("Inconsistent ndim and "
                "underlying_stride.size()");
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
    if(tmp != traits.nelems)
    {
        throw std::runtime_error("Inconsistent nelems");
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
    for(size_t i = 0; i < traits.ndim; ++i)
    {
        if(traits.offset[i] >= traits.underlying_shape[i])
        {
            throw std::runtime_error("Inconsistent offset and "
                    "underlying_shape");
        }
        if(traits.offset[i]+traits.shape[i] > traits.underlying_shape[i])
        {
            throw std::runtime_error("Inconsistent offset, shape and "
                    "underlying_shape");
        }
    }
    tmp = 1;
    for(size_t i = 0; i < traits.ndim; ++i)
    {
        if(traits.underlying_stride[i] != tmp)
        {
            throw std::runtime_error("Inconsistent underlying_stride");
        }
        tmp *= traits.underlying_shape[i];
    }
    std::vector<size_t> last_index(traits.offset);
    for(size_t i = 0; i < traits.ndim; ++i)
    {
        last_index[i] += traits.shape[i];
    }
    for(size_t i = 0; i < traits.ndim; ++i)
    {
        std::vector<size_t> index(last_index);
        ++index[i];
        TESTN(traits.index_to_linear(index));
        TESTA(traits.contains_index(index) == 0);
    }
    TESTN(traits.linear_to_index(traits.nelems));
    for(size_t i = 0; i < traits.nelems; ++i)
    {
        auto index = traits.linear_to_index(i);
        TESTA(traits.contains_index(index) == 1);
        TESTA(i == traits.index_to_linear(index));
    }
}

int main(int argc, char **argv)
{
    // Check scalar case with empty shape
    TileTraits scalar_traits({});
    TESTP(TileTraits({}, {}, {}));
    TESTN(TileTraits({}, {1}, {}));
    TESTN(TileTraits({}, {}, {1}));
    TESTA(scalar_traits.shape == std::vector<size_t>());
    std::cout << "scalar case\n" << scalar_traits;
    validate_traits(scalar_traits);
    TESTA((scalar_traits.index_to_linear({}) == 0));
    TESTN((scalar_traits.index_to_linear({0})));
    // Check vector case
    TileTraits vector_traits({10});
    TESTN(TileTraits({0}));
    TESTP(TileTraits({10}, {9}, {19}));
    TESTN(TileTraits({10}, {10}, {19}));
    std::cout << "vector case\n" << vector_traits;
    validate_traits(vector_traits);
    TESTA((vector_traits.index_to_linear({5}) == 5));
    TESTN((vector_traits.index_to_linear({10})));
    TESTN((vector_traits.index_to_linear({0, 0})));
    TESTN((vector_traits.index_to_linear({})));
    // Check matrix case
    TileTraits matrix_traits({3, 5});
    TESTN(TileTraits({3, 0}));
    std::cout << "matrix case\n" << matrix_traits;
    validate_traits(matrix_traits);
    TESTA((matrix_traits.index_to_linear({1, 1}) == 4));
    TESTN((matrix_traits.index_to_linear({3, 2})));
    TESTN((matrix_traits.index_to_linear({2, 5})));
    TESTN((matrix_traits.index_to_linear({1})));
    TESTN((matrix_traits.index_to_linear({1, 1, 1})));
    // Check other matrix case
    TileTraits submatrix_traits({3, 5}, {3, 5}, {10, 10});
    validate_traits(submatrix_traits);
    submatrix_traits.index_to_linear({3, 5});
    submatrix_traits.index_to_linear({5, 9});
    TESTN(submatrix_traits.index_to_linear({5, 10}));
    TESTN(submatrix_traits.index_to_linear({2, 5}));
    TESTN(submatrix_traits.index_to_linear({3, 4}));
    validate_traits(TileTraits({1, 1, 2}));
    TESTN((TileTraits({1, 1, 2}).index_to_linear({0, 1, 0})));
    // Check 5-dimensional tensor
    TileTraits t5d_traits({7, 9, 11, 13, 15});
    TESTN(TileTraits({7, 9, 11, 13, 0}));
    TESTA(t5d_traits.shape == std::vector<size_t>({7, 9, 11, 13, 15}));
    std::cout << "5D-tensor case\n" << t5d_traits;
    validate_traits(t5d_traits);
    TESTA((t5d_traits.index_to_linear({0, 1, 2, 3, 4}) == 38248));
    TESTN((t5d_traits.index_to_linear({0, 1, 2, 3, 15})));
    TESTN((t5d_traits.index_to_linear({0, 1, 2, 3})));
    TESTN((t5d_traits.index_to_linear({0, 1, 2, 3, 4, 5})));
    // Check constructor from vector
    TileTraits traits(std::vector<size_t>{1, 2, 3, 4});
    validate_traits(traits);
    return 0;
}

