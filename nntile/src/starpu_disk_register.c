/*! @file src/starpu_disk_register.c
 * C linkage for starpu_disk_register.
 *
 * StarPU 1.4.3 includes starpu_disk.h before extern "C" in starpu.h, so a
 * C++ call site looks up a mangled symbol that libstarpu does not export.
 */

#include <starpu.h>

int nntile_starpu_disk_register(
    struct starpu_disk_ops *ops,
    void *parameter,
    starpu_ssize_t size)
{
    return starpu_disk_register(ops, parameter, size);
}
