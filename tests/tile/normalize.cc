#include "nntile/tile/normalize.hh"
#include "../testing.hh"

using namespace nntile;

Starpu starpu;

template<typename T>
void validate_normalize()
{
    Starpu::pause();
    TileTraits traits({4, 5, 6, 7});
    std::vector<T> data(traits.nelems);
    for(Index i = 0; i < traits.nelems; ++i)
    {
        data[i] = T{1};
    }
    T gamma_beta[2] = {T{2}, T{1}};
    StarpuVariableHandle gamma_beta_handle(
            reinterpret_cast<uintptr_t>(gamma_beta), sizeof(gamma_beta));
    Tile<T> A(traits, &data[0], traits.nelems);
    for(Index axis = 0; axis < A.ndim; ++axis)
    {
        std::vector<Index> B_shape(traits.shape);
        B_shape[0] = 2;
        for(Index i = 0; i < axis ; ++i)
        {
            B_shape[i+1] = traits.shape[i];
        }
        Tile<T> B(B_shape);
        auto B_local = B.acquire(STARPU_W);
        for(Index i = 0; i < B.nelems; ++i)
        {
            B_local[i] = T{1};
        }
        B_local.release();
        Starpu::resume();
        normalize(gamma_beta_handle, B, A, A.shape[axis], T{0}, axis);
        Starpu::pause();
    }
    Starpu::resume();
}

int main(int argc, char **argv)
{
    validate_normalize<fp32_t>();
    validate_normalize<fp64_t>();
    return 0;
}

