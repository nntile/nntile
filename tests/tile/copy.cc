#include "nntile/tile/copy.hh"
#include "nntile/tile/randn.hh"

using namespace nntile;

template<typename T>
void check_copy(const Tile<T> &src, const std::vector<size_t> src_coord,
        const Tile<T> &dst, const std::vector<size_t> dst_coord)
{
    copy(src, src_coord, dst, dst_coord);
    size_t ndim = src.ndim;
    const auto src_ptr = src.get_local_ptr(), dst_ptr = dst.get_local_ptr();
    std::vector<size_t> src_index(ndim), dst_index(ndim, 0);
    bool ignore_element = false;
    for(size_t k = 0; k < ndim; ++k)
    {
        size_t global_coord = dst_coord[k];
        if((global_coord >= src_coord[k]+src.shape[k])
                or (global_coord < src_coord[k]))
        {
            ignore_element = true;
            break;
        }
        src_index[k] = global_coord - src_coord[k];
    }
    if(!ignore_element)
    {
        size_t src_offset = src_index[0];
        for(size_t k = 1; k < ndim; ++k)
        {
            src_offset += src_index[k] * src.stride[k];
        }
        if(dst_ptr[0] != src_ptr[src_offset])
        {
            throw std::runtime_error("dst_ptr[0] != src_ptr[src_offset]");
        }
    }
    for(size_t i = 1; i < dst.nelems; ++i)
    {
        ++dst_index[0];
        size_t j = 0;
        while(dst_index[j] == dst.shape[j])
        {
            dst_index[j] = 0;
            ++j;
            ++dst_index[j];
        }
        ignore_element = false;
        for(size_t k = 0; k < ndim; ++k)
        {
            size_t global_coord = dst_index[k] + dst_coord[k];
            if((global_coord >= src_coord[k]+src.shape[k])
                    or (global_coord < src_coord[k]))
            {
                ignore_element = true;
                break;
            }
            src_index[k] = global_coord - src_coord[k];
        }
        if(!ignore_element)
        {
            size_t src_offset = src_index[0];
            for(size_t k = 1; k < ndim; ++k)
            {
                src_offset += src_index[k] * src.stride[k];
            }
            if(dst_ptr[i] != src_ptr[src_offset])
            {
                throw std::runtime_error("dst_ptr[i] != src_ptr[src_offset]");
            }
        }
    }
}

template<typename T>
void validate_copy()
{
    Tile<T> A({4, 5, 6, 7}), B({2, 3, 4, 5});
    unsigned long long seed = 100;
    T one = 1, zero = 0;
    randn(A, std::vector<size_t>(A.ndim, 0), A.stride, seed, zero, one);
    check_copy(A, {0, 0, 0, 0}, B, {0, 0, 0, 0});
    check_copy(A, {1, 2, 3, 4}, B, {2, 3, 4, 5});
    check_copy(A, {1, 2, 3, 4}, B, {0, 0, 2, 2});
    check_copy(A, {1, 2, 3, 4}, B, {4, 5, 8, 0});
    check_copy(A, {1, 2, 3, 4}, B, {4, 5, 8, 11});
}

int main(int argc, char **argv)
{
    StarPU starpu;
    validate_copy<float>();
    validate_copy<double>();
    return 0;
}

