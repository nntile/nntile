/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/config.hh
 * Base configuration of NNTile with its initialization
 *
 * @version 1.1.0
 * */

#pragma once

#include <starpu.h>

namespace nntile
{

class Config: public starpu_conf
{
public:
    Config()
    {
    }
    void init(int &argc, char **&argv);
    void shutdown();
};

extern Config config;

} // namespace nntile
