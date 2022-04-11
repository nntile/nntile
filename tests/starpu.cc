#include <nntile/starpu.hh>
#include <iostream>

using namespace nntile;

template<typename T>
void test()
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
    StarPU conf;
    test<float>();
    return 0;
}

