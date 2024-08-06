from typing import Literal

class TransOp:
    def __init__(self, value_: Literal[0, 1]) -> None: ...

notrans: TransOp
trans: TransOp
