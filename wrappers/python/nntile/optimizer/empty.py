# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/optimizer/empty.py
# This optimizer does nothing, it does not update parameters at all
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-09-15

import nntile
import numpy as np
from nntile.tensor import TensorTraits

class Empty:
    def __init__(self, params, next_tag):
        self.params = params
        self.next_tag = next_tag

    def get_next_tag(self):
        return self.next_tag
    
    def unregister(self):
        pass

    def step(self):
        pass
