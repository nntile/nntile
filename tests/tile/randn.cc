#include <nntile/tile/randn.hh>
#include <iostream>

using namespace nntile;

template<typename T>
void compare(const Tile<T> &A, const Tile<T> &B,
        const std::vector<size_t> &offset)
{
    std::vector<size_t> index(A.ndim, 0);
    size_t offset_A = 0, offset_B = offset[0];
    for(size_t i = 1; i < A.ndim; ++i)
    {
        offset_B += offset[i] * B.stride[i];
    }
    const T *ptr_A = A.get_local_ptr(), *ptr_B = B.get_local_ptr();
    if(ptr_A[0] != ptr_B[offset_B])
    {
        throw std::runtime_error("ptr_A[0] != ptr_B[offset_B]");
    }
    for(size_t i = 1; i < A.nelems; ++i)
    {
        ++offset_A;
        size_t j = 0;
        ++index[0];
        while(index[j] == A.shape[j])
        {
            index[j] = 0;
            ++j;
            ++index[j];
        }
        offset_B = index[0] + offset[0];
        for(size_t k = 1; k < A.ndim; ++k)
        {
            offset_B += (index[k]+offset[k]) * B.stride[k];
        }
        if(ptr_A[offset_A] != ptr_B[offset_B])
        {
            throw std::runtime_error("ptr_A[offset_A] != ptr_B[offset_B]");
        }
    }
}

template<typename T>
void validate_randn()
{
    Tile<T> big({4, 4, 4, 4}), big2({4, 4, 4, 4}), small({2, 2, 2, 2});
    T one = 1, zero = 0;
    unsigned long long seed = 100, seed2 = seed, seed3 = seed;
    std::vector<size_t> offset({1, 1, 2, 2});
    randn(big, {0, 0, 0, 0}, big.stride, seed, zero, one);
    randn(big2, {0, 0, 0, 0}, big.stride, seed2, zero, one);
    randn(small, offset, big.stride, seed3, zero, one);
    compare(big2, big, {0, 0, 0, 0});
    compare(small, big, offset);
}

int main(int argc, char **argv)
{
    StarPU starpu;
    validate_randn<float>();
    validate_randn<double>();
    return 0;
}

