/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/grad_mode.cc
 * GradMode implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/grad_mode.hh"

namespace nntile::graph
{

namespace
{
thread_local bool g_grad_enabled = true;
}

bool GradMode::is_enabled()
{
    return g_grad_enabled;
}

void GradMode::set_enabled(bool enabled)
{
    g_grad_enabled = enabled;
}

GradMode::Guard::Guard() : prev_(g_grad_enabled)
{
    g_grad_enabled = false;
}

GradMode::Guard::~Guard()
{
    g_grad_enabled = prev_;
}

} // namespace nntile::graph
