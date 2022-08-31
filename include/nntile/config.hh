/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/config.hh
 * Base configuration of NNTile with its initialization
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-31
 * */

#pragma once

#include <starpu.h>

namespace nntile
{

class Config: public starpu_conf
{
public:
    void init(int &argc, char **&argv);
    void shutdown();
};

extern Config config;

} // namespace nntile

