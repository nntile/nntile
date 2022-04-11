#pragma once

#include <nntile/tile/tile.hh>
#include <cmath>

namespace nntile
{

template<typename T>
void gelu_codelet_cpu(void *buffers[], void *cl_args)
{
    size_t nelems;
    starpu_codelet_unpack_args(cl_args, &nelems);
    T *data = reinterpret_cast<T *>(STARPU_VECTOR_GET_PTR(buffers[0]));
    constexpr T sqrt2 = sqrt(T{2.0});
    constexpr T one = 1.0;
    constexpr T pt5 = 0.5;
    for(size_t i = 0; i < nelems; ++i)
    {
        T tmp = pt5*(std::erf(data[i]/sqrt2)) + pt5;
        data[i] *= tmp;
    }
}

template<typename T>
void gelu_async(const Tile<T> &A)
{
    static struct starpu_codelet codelet_gelu =
    {
        .cpu_funcs = {gelu_codelet_cpu<T>},
        .nbuffers = 1,
        .modes = {STARPU_RW}
    };
    size_t nelems = A.nelems;
    starpu_task_insert(&codelet_gelu,
            STARPU_VALUE, &nelems, sizeof(nelems),
            STARPU_RW, static_cast<starpu_data_handle_t>(A),
            0);
}

template<typename T>
void gelu(const Tile<T> &A)
{
    gelu_async<T>(A);
    starpu_task_wait_for_all();
}

} // namespace nntile

