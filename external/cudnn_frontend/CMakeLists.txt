# Define version of CUDNN frontend to acquire
set(CUDNN_FRONTEND_VERSION 1.11.0)

# Declare cudnn frontend to be acquired
FetchContent_Declare(
    cudnn_frontend
    URL https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v${CUDNN_FRONTEND_VERSION}.tar.gz
    URL_HASH SHA256=de723e07582603721c97dd4fd82bbad0020fa830d1d1eee6269ebf6950ed3128
)

# Acquire cudnn frontend
FetchContent_MakeAvailable(cudnn_frontend)
