#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::lion_step
{

template<typename T>
void cuda(cudaStream_t stream,
          Index num_iter,
          Index num_elems,
          Scalar beta_1,
          Scalar beta_2,
          Scalar lr,
          Scalar weight_decay,
          const T *grad,
          T *first_moment,
          T *p)
    noexcept;

} // namespace nntile::kernel::lion_step
