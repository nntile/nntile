from typing import Sequence

def block_cyclic(tensor_grid: Sequence[int], mpi_grid: Sequence[int],
                 start_rank: int, max_rank: int) -> list[int]: ...
