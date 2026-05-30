# Python examples

Graph API examples mirroring [nntile/examples/](../../nntile/examples/) C++ programs.

## Build

From the repository root:

```bash
cmake -S . -B build -GNinja -DCMAKE_CXX_COMPILER=g++ -DUSE_CUDA=OFF
cmake --build build --target nntile_py
export PYTHONPATH="$(pwd)/build/python"
export LD_LIBRARY_PATH="$(pwd)/build/nntile:/opt/starpu/lib:${LD_LIBRARY_PATH}"
```

## Scripts

| Python | C++ source | Description |
|--------|------------|-------------|
| [mlp_example.py](mlp_example.py) | [graph_mlp_example.cc](../../nntile/examples/graph_mlp_example.cc) | MLP forward/backward smoke test |
| [gpt2_training.py](gpt2_training.py) | [gpt2_graph_training.cc](../../nntile/examples/gpt2_graph_training.cc) | Tiny GPT-2 training on `train.bin` |

## GPT-2 tiny demo

```bash
python3 nntile/examples/prepare_tiny_train_bin.py \
  --output /tmp/nntile_demo/train.bin --seq-len 8 --batch-size 2 \
  --num-batches 4 --vocab-size 256

python3 python/examples/gpt2_training.py \
  --train-bin /tmp/nntile_demo/train.bin --tiny --epochs 1 --max-batches 1
```

Multi-batch incremental training (`--max-batches` > 1) matches the C++ loop structure; additional
bind-hint / phase wiring may be required for batch 2+ in this Python v1.
