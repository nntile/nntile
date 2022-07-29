#include "nntile/tensor/sumnorm.hh"
#include "nntile/tile/sumnorm.hh"
#include "nntile/tensor/randn.hh"
#include "nntile/tensor/copy.hh"
#include "nntile/tensor/clear.hh"
#include "nntile/tile/clear.hh"
#include "../testing.hh"

using namespace nntile;

Starpu starpu;

template<typename T>
void check_sumnorm(const Tensor<T> &src, const Tile<T> &src_tile, Index axis)
{
    std::vector<Index> shape(src.ndim), basetile(src.ndim);
    shape[0] = 2;
    basetile[0] = 2;
    for(Index i = 0; i < axis; ++i)
    {
        shape[i+1] = src.shape[i];
        basetile[i+1] = src.basetile_shape[i];
    }
    for(Index i = axis+1; i < src.ndim; ++i)
    {
        shape[i] = src.shape[i];
        basetile[i] = src.basetile_shape[i];
    }
    Tensor<T> dst(shape, basetile);
    Tile<T> tile(shape), tile2(shape);
    clear_async(dst);
    clear_async(tile);
    sumnorm_async(src, dst, axis);
    sumnorm_async(src_tile, tile, axis);
    copy(dst, tile2);
    Starpu::pause();
    auto local = tile.acquire(STARPU_R),
         local2 = tile2.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; i += 2)
    {
        T sum = local[i], sum2 = local2[i];
        T diff = std::abs(sum-sum2), abs = std::abs(sum);
        T threshold = 50 * abs * std::numeric_limits<T>::epsilon();
        if(diff > threshold)
        {
            std::cout << diff << " " << threshold << "\n";
            std::cout << sum << " " << sum2 << "\n";
            throw std::runtime_error("Invalid sum");
        }
        T norm = local[i+1], norm2 = local2[i+1];
        diff = std::abs(norm-norm2);
        abs = std::abs(norm);
        threshold = 50 * abs * std::numeric_limits<T>::epsilon();
        if(diff > threshold)
        {
            std::cout << diff << " " << threshold << "\n";
            std::cout << norm << " " << norm2 << "\n";
            throw std::runtime_error("Invalid norm");
        }
    }
    Starpu::resume();
}

template<typename T>
void validate_sumnorm()
{
    Tensor<T> A({9, 10, 13, 15}, {4, 5, 6, 7});
    Tile<T> A_tile(A.shape);
    constexpr unsigned long long seed = 100000000000001ULL;
    // Avoid mean=0 because of instable relative error of sum (division by 0)
    randn(A, seed, T{1}, T{1});
    copy(A, A_tile);
    for(Index i = 0; i < A.ndim; ++i)
    {
        check_sumnorm(A, A_tile, i);
    }
}

int main(int argc, char **argv)
{
    validate_sumnorm<fp32_t>();
    validate_sumnorm<fp64_t>();
    return 0;
}

