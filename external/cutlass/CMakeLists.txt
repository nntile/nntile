set(CUTLASS_VERSION 3.8.0)

FetchContent_Declare(
    cutlass
    URL https://github.com/NVIDIA/cutlass/archive/refs/tags/v${CUTLASS_VERSION}.zip
    URL_HASH SHA256=a7739ca3dc74e3a5cb57f93fc95224c5e2a3c2dff2c16bb09a5e459463604c08
    FIND_PACKAGE_ARGS GLOBAL NAMES cutlass
)
