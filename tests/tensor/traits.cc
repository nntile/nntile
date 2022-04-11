#include <nntile/tensor/traits.hh>

using namespace nntile;

void validate_traits()
{
    std::vector<size_t> shape({1000, 11, 12, 1300}),
        tile_shape({512, 4, 4, 512});
    TensorTraits A_traits(shape, tile_shape);
    std::cout << A_traits;
}

int main(int argc, char **argv)
{
    validate_traits();
    return 0;
}

