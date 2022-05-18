#include "nntile/tile/tile.hh"
#include "nntile/base_types.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void check_tile(const std::vector<Index> &shape)
{
    // Check tile with allocation done by StarPU
    Tile<T> tile1(shape);
    TESTA(static_cast<starpu_data_handle_t>(tile1) != nullptr);
    auto tile1_local = tile1.acquire(STARPU_W);
    TESTA(tile1_local.get_ptr() != nullptr);
    for(Index i = 0; i < tile1.nelems; ++i)
    {
        tile1_local[i] = static_cast<T>(i);
    }
    tile1_local.release();
    // Check copy construction
    Tile<T> tile2(tile1);
    TESTA(static_cast<starpu_data_handle_t>(tile2) ==
            static_cast<starpu_data_handle_t>(tile1));
    auto tile2_local = tile2.acquire(STARPU_R);
    for(Index i = 0; i < tile1.nelems; ++i)
    {
        TESTA(tile2_local[i] == static_cast<T>(i));
    }
    tile2_local.release();
    // Check constructor from TileTraits
    Tile<T> tile3(static_cast<TileTraits>(tile2));
    TESTA(static_cast<starpu_data_handle_t>(tile3) != nullptr);
    TESTA(static_cast<starpu_data_handle_t>(tile2) !=
            static_cast<starpu_data_handle_t>(tile3));
    // Check with shape and pointer
    std::vector<T> data(tile1.nelems);
    for(Index i = 0; i < tile1.nelems; ++i)
    {
        data[i] = static_cast<T>(i);
    }
    TESTN(Tile<T>(shape, &data[0], tile1.nelems-1));
    Tile<T> tile4(tile1, &data[0], tile1.nelems);
    TESTA(static_cast<starpu_data_handle_t>(tile4) != nullptr);
    auto tile4_local = tile4.acquire(STARPU_R);
    for(Index i = 0; i < tile1.nelems; ++i)
    {
        TESTA(tile4_local[i] == static_cast<T>(i));
    }
    tile4_local.release();
    // Check with TileTraits and pointer
    TESTN(Tile<T>(tile4, &data[0], tile4.nelems-1));
    Tile<T> tile5(tile4, &data[0], tile4.nelems);
    TESTA(static_cast<starpu_data_handle_t>(tile5) != nullptr);
    TESTA(static_cast<starpu_data_handle_t>(tile5) !=
            static_cast<starpu_data_handle_t>(tile4));
    auto tile5_local = tile5.acquire(STARPU_RW);
    for(Index i = 0; i < tile1.nelems; ++i)
    {
        TESTA(tile5_local[i] == static_cast<T>(i));
        tile5_local[i] += T{1};
    }
    tile5_local.release();
    // Check if 2 handles with the same user-provided buffer are actually
    // sharing data
    tile4_local.acquire(STARPU_R);
    for(Index i = 0; i < tile1.nelems; ++i)
    {
        TESTA(tile4_local[i] == static_cast<T>(i+1));
    }
    tile4_local.release();
}

template<typename T>
void validate_tile()
{
    check_tile<T>({});
    check_tile<T>({3});
    check_tile<T>({3, 2});
    check_tile<T>({3, 2, 1});
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_tile<fp64_t>();
    validate_tile<fp32_t>();
    return 0;
}

