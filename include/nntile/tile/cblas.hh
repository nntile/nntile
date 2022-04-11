#pragma once

//#define NNTILE_USE_APPLE_ACCELERATE

#if defined(NNTILE_USE_APPLE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#   define CBLAS_INT int
#elif defined(NNTILE_USE_INTEL_MKL)
#   include <mkl.h>
#else
#   include <cblas.h>
#endif

// Define type CBLAS_INT to use cblas properly
#ifndef CBLAS_INT
#   if defined(f77_int)
#      define CBLAS_INT f77_int
#   elif defined(CBLAS_INDEX)
#       define CBLAS_INT CBLAS_INDEX
#   endif
#endif

