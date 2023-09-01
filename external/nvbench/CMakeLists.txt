set(NVBENCH_COMMIT "f57aa9c993f4392a76650bc54513f571cd1128c9")

FetchContent_Declare(
    nvbench
    URL https://github.com/NVIDIA/nvbench/archive/${NVBENCH_COMMIT}.tar.gz
    URL_HASH SHA256=f23caf8e94d231b40fcfffec4cdf4ce469e6dcdd5a7e9f4d189fde4f8e2dd266
    FIND_PACKAGE_ARGS GLOBAL NAMES nvbench
)
