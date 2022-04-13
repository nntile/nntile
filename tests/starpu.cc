#include "nntile/starpu.hh"
#include "testing.hh"
#include "testing.hh"
#include <iostream>
#include <cstdlib>

using namespace nntile;

template<typename T>
void test_starpu()
{
    StarpuHandle empty;
    StarpuHandle *x = new StarpuVariableHandle(100*sizeof(T));
    delete x;
    T data[100];
    uintptr_t ptr = reinterpret_cast<uintptr_t>(data);
    StarpuVariableHandle y(ptr, 100*sizeof(T));
    StarpuVariableHandle z(y);
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

