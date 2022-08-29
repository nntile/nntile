/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/common.hh
 * Init and shutdown StarPU for testing
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-23
 * */

#pragma once

#include <starpu.h>
#include <nntile/defs.h>

class StarpuTest
{
    starpu_conf conf;
public:
    StarpuTest()
    {
        // Init StarPU configuration at first
        starpu_conf conf;
        int ret = starpu_conf_init(&conf);
        if(ret != 0)
        {
            throw std::runtime_error("starpu_conf_init error");
        }
        // Set number of workers to 1 where applicable
        conf.ncpus = 1;
#ifdef NNTILE_USE_CUDA
        conf.ncuda = 1;
#else // NNTILE_USE_CUDA
        conf.ncuda = 0;
#endif // NNTILE_USE_CUDA
        conf.nopencl = 0;
        // Set history-based scheduler to check perfmodel footprint usage
        conf.sched_policy_name = "dmda";
        // Init StarPU with the config
        ret = starpu_init(&conf);
        if(ret != 0)
        {
            throw std::runtime_error("starpu_init error");
        }
        // Pause StarPU, as it will be enabled only for when needed
        starpu_pause();
    }
    ~StarpuTest()
    {
        // Resume StarPU and shut it down
        starpu_resume();
        starpu_shutdown();
    }
};

