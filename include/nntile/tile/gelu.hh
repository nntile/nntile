#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

template<typename T>
void gelu(ContiguousTile<T> &A)
{
    gelu_async(A);
    starpu_task_wait_for_all();
}

template<typename T>
void gelu_codelet_cpu(void *buffers[], void *cl_args)
{
    int nelems = STARPU_VECTOR_GET_NX(buffers[0]);
    T *data = reinterpret_cast<T *>(STARPU_VECTOR_GET_PTR(buffers[0]));
    constexpr T sqrt2 = sqrt(T{2.0});
    constexpr T one = 1.0;
    constexpr T pt5 = 0.5;
    for(int i = 0; i < nelems; ++i)
    {
        T tmp = pt5*(std::erf(data[i]/sqrt2)) + pt5;
        data[i] *= tmp;
    }
}

template<typename T>
void gelu_async(ContiguousTile<T> &A)
{
    static struct starpu_codelet codelet_gelu =
    {
        .cpu_funcs = {gelu_codelet_cpu<T>},
        .nbuffers = 1,
        .modes = {STARPU_RW}
    };
    starpu_task_insert(&codelet_gelu,
            STARPU_RW, A.handle,
            0);
}

} // namespace nntile

