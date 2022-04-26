NNTile
======

**NNTile** is a software for training big neural networks over heterogeneous distributed-memory systems. The approach is based on divide-and-conquer principle: all data are stored in small chunks, so-called tiles, and each layer is applied directly to the tiles. This way each layer, as well as entire neural network, is represented by a directed acyclic graph (DAG for short) of tasks operating on separate tiles. Execution of tasks is scheduled dynamically with help of StarPU library. The StarPU library is meant for distributed heterogeneous computing and allows using different processing units on the same node.
