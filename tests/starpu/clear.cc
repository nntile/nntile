/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/clear.cc
 * Clear a StarPU buffer
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/config.hh"
#include "nntile/starpu/config.hh"
#include "nntile/starpu/clear.hh"
#include "../testing.hh"
#ifdef NNTILE_USE_CUDA
#   include <cuda_runtime.h>
#endif // NNTILE_USE_CUDA
#include <vector>
#include <stdexcept>
#include <iostream>

using namespace nntile;
using namespace nntile::starpu;

void validate(std::size_t size)
{
    // Init all the data
    std::vector<char> data_init(size);
    for(std::size_t i = 0; i < size; ++i)
    {
        data_init[i] = -1;
    }
    // Create copy of data
    std::vector<char> data(data_init);
    // Check by actually submitting a task
    VariableHandle data_handle(&data[0], size);
    clear::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::clear::submit restricted to CPU\n";
    clear::submit(data_handle);
    starpu_task_wait_for_all();
    data_handle.unregister();
    // Check result
    for(std::size_t i = 0; i < size; ++i)
    {
        TEST_ASSERT(data[i] == 0);
    }
    std::cout << "OK: starpu::clear::submit restricted to CPU\n";
#ifdef NNTILE_USE_CUDA
    // Check by actually submitting a task
    data = data_init;
    data_handle = VariableHandle(&data[0], size);
    clear::restrict_where(STARPU_CUDA);
    std::cout << "Run starpu::clear::submit restricted to CUDA\n";
    clear::submit(data_handle);
    starpu_task_wait_for_all();
    data_handle.unregister();
    // Check result
    for(std::size_t i = 0; i < size; ++i)
    {
        TEST_ASSERT(data[i] == 0);
    }
    std::cout << "OK: starpu::clear::submit restricted to CUDA\n";
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    // Initialize StarPU (it will automatically shutdown itself on exit)
    int ncpus=1, ncuda=1, cublas=0, ooc=0, ooc_disk_node_id=-1, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto config = starpu::Config(
        ncpus, ncuda, cublas, ooc, ooc_path, ooc_size, ooc_disk_node_id, verbose
    );

    // Launch all tests
    validate(1);
    validate(100000);

    return 0;
}
