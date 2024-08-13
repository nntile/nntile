# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/optimizer/empty.py
# This optimizer does nothing, it does not update parameters at all
#
# @version 1.1.0

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

    def get_nbytes(self):
        return 0
