# Define version of CUDNN frontend to acquire
set(CUDNN_FRONTEND_VERSION 1.18.0)

# Declare cudnn frontend to be acquired
FetchContent_Declare(
    cudnn_frontend
    URL https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v${CUDNN_FRONTEND_VERSION}.tar.gz
    URL_HASH SHA256=122dde9e9ee37cc05d626097b3c85d507be0db3f73ee196baa5a034069460517
)

# Disable cudnn frontend samples and tests
set(CUDNN_FRONTEND_BUILD_SAMPLES OFF)
set(CUDNN_FRONTEND_BUILD_TESTS OFF)

# Acquire cudnn frontend
FetchContent_MakeAvailable(cudnn_frontend)
