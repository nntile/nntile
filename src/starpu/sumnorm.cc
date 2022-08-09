

//! Sum and Euclidian norm along middle axis of StarPU buffer
//
// See sumnorm_kernel_cpu function for more info.
//
// @sa sumnorm_kernel_cpu, clear_starpu_cpu
template<typename T>
void sumnorm_starpu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<sumnorm_starpu_args *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    // Launch kernel
    const T *src = interfaces[0]->get_ptr<T>();
    T *sumnorm = interfaces[1]->get_ptr<T>();
    sumnorm_kernel_cpu<T>(args->m, args->n, args->k, src, sumnorm);
}

// Explicit instantiation
template
void sumnorm_starpu_cpu<fp32_t>(void *buffers[], void *cl_args)
    noexcept;

template
void sumnorm_starpu_cpu<fp64_t>(void *buffers[], void *cl_args)
    noexcept;
