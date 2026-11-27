/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/config.cc
 * Test for NNTile configuration object
 *
 * @version 1.1.0
 * */

#include "nntile/starpu.hh"
#include "testing.hh"
#include <iostream>
#include <cstdlib>
#include <vector>

using namespace nntile;

template<typename T>
void test_starpu()
{
    StarpuHandle *x = new StarpuVariableHandle(100*sizeof(T));
    delete x;
    std::vector<T> data(100);
    uintptr_t ptr = reinterpret_cast<uintptr_t>(&data[0]);
    StarpuVariableHandle y(ptr, 100*sizeof(T));
    StarpuVariableHandle z(y);
    auto y_local = y.acquire(STARPU_R);
    TESTA(y_local.get_ptr() == &data[0]);
    y_local.release();
    auto z_local = z.acquire(STARPU_RW);
    TESTA(z_local.get_ptr() == &data[0]);
    z_local.release();
    // All the local handles/buffers shall be released before syncing to avoid
    // dead lock
    starpu_task_wait_for_all();
}

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        throw std::runtime_error("Execute this test as \"starpu test_index\"");
    }
    int test = std::atoi(argv[1]);
    if(test == 0)
    {
        throw std::runtime_error("Could not convert argument to int");
    }
    if(test == 1)
    {
        struct starpu_conf conf;
        conf.magic = 41;
        TESTN(Starpu starpu(conf));
    }
    else if(test == 2)
    {
        Starpu starpu;
    }
    else if(test == 3)
    {
        Starpu starpu;
        TESTP(Starpu starpu2);
    }
    else if(test == 4)
    {
        Starpu starpu;
        test_starpu<float>();
        test_starpu<double>();
    }
    else if(test == 5)
    {
        void *arg_buffer;
        size_t arg_buffer_size;
        constexpr char cval = 127;
        constexpr int ival = 10;
        constexpr long lval = 20;
        constexpr float fval = -0.4;
        constexpr double dval = 0.4;
        constexpr int64_t Indval[4] = {4, 6, 8, 12};
        starpu_codelet_pack_args(&arg_buffer, &arg_buffer_size,
                STARPU_VALUE, &cval, sizeof(cval),
                STARPU_VALUE, &ival, sizeof(ival),
                STARPU_VALUE, &lval, sizeof(lval),
                STARPU_VALUE, &fval, sizeof(fval),
                STARPU_VALUE, &dval, sizeof(dval),
                STARPU_VALUE, &Indval, sizeof(Indval),
                0);
        const char *cptr = nullptr;
        const int *iptr = nullptr;
        const long *lptr = nullptr;
        const float *fptr = nullptr;
        const double *dptr = nullptr;
        const int64_t *Indptr = nullptr;
        Starpu::unpack_args_ptr(arg_buffer, cptr, iptr, lptr, fptr, dptr,
                Indptr);
        TESTA(cptr and cptr[0] == cval);
        TESTA(iptr and iptr[0] == ival);
        TESTA(lptr and lptr[0] == lval);
        TESTA(fptr and fptr[0] == fval);
        TESTA(dptr and dptr[0] == dval);
        TESTA(Indptr and Indptr[0] == Indval[0]);
        TESTA(Indptr[1] == Indval[1]);
        TESTA(Indptr[2] == Indval[2]);
        TESTA(Indptr[3] == Indval[3]);
        // Check if a value behind last packed arg is not set
        const void *voidptr = nullptr;
        Starpu::unpack_args_ptr(arg_buffer, cptr, iptr, lptr, fptr, dptr,
                Indptr, voidptr);
        TESTA(!voidptr);
        // Check partial read
        cptr = nullptr;
        Starpu::unpack_args_ptr(arg_buffer, cptr);
        TESTA(cptr and cptr[0] == cval);
    }
    else
    {
        throw std::runtime_error("Invalid test index");
    }
    return 0;
}
