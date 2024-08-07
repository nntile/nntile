class Config:
    def __init__(self, ncpus: int = ..., ncuda: int = ..., cublas: int = ...,
                 logger_server_addr: str = ...,
                 logger_server_port: int = ...) -> None: ...
    def shutdown(self) -> None: ...

def init() -> None: ...
def pause() -> None: ...
def resume() -> None: ...
def wait_for_all() -> None: ...

def profiling_disable() -> None: ...
def profiling_enable() -> None: ...
def profiling_init() -> None: ...

def restrict_cpu() -> None: ...
def restrict_cuda() -> None: ...
def restrict_restore() -> None: ...
