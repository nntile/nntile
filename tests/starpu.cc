#include "nntile/starpu.hh"
#include "testing.hh"
#include <iostream>
#include <cstdlib>

using namespace nntile;

template<typename T>
void test_starpu()
{
    StarpuHandle *x = new StarpuVariableHandle(100*sizeof(T));
    x->invalidate();
    x->wont_use();
    delete x;
    T data[100];
    uintptr_t ptr = reinterpret_cast<uintptr_t>(data);
    void *ptr_void = reinterpret_cast<void *>(data);
    StarpuVariableHandle y(ptr, 100*sizeof(T));
    y.acquire(STARPU_RW);
    TESTN(y.acquire(static_cast<enum starpu_data_access_mode>(-1)));
    TESTN(y.acquire(STARPU_ACCESS_MODE_MAX));
    y.release();
    StarpuVariableHandle z(y);
    TESTA((z.get_local_ptr() == ptr_void));
    y.invalidate_submit();
    starpu_task_wait_for_all();
    TESTA(starpu_data_get_local_ptr(static_cast<starpu_data_handle_t>(y))
            == ptr_void);
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
        TESTN(Starpu starpu2);
    }
    else if(test == 4)
    {
        Starpu starpu;
        test_starpu<float>();
        test_starpu<double>();
    }
    else
    {
        throw std::runtime_error("Invalid test index");
    }
    return 0;
}

