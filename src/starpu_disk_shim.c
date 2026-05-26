/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/starpu_disk_shim.c
 * C linkage wrapper for StarPU disk registration (starpu_disk.h is included
 * before extern "C" in starpu.h, so C++ TU gets the wrong linkage).
 *
 * @version 1.1.0
 * */

#include <starpu_disk.h>

int nntile_starpu_disk_register_unistd(void *parameter, starpu_ssize_t size)
{
    return starpu_disk_register(&starpu_disk_unistd_ops, parameter, size);
}
