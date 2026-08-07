python causal_lm_data_preparation.py --seq-len=256 --batch-size=1 --dataset-select=6

STARPU_SILENT=1 STARPU_NCUDA=4 STARPU_NCPU=1 python llama_training.py --restrict="cuda" --pretrained=local --config-path="llama_config_100b.json" --save-checkpoint-path="" --optimizer="sgd" --seq-len=256 --lr=1e-5 --dtype=bf16 --nepochs=1 --batch-size=1 --minibatch-size=1 --dataset-file="tinystories/train.bin" --hidden-size-tile=7710 --intermediate-size-tile=30840 --n-head-tile=1

# Number of parameters: 100045191300
# Converting PyTorch model to NNTile requires 3182.811262369156 seconds
# From PyTorch loader to NNTile batches in 0.0004851818084716797 seconds
# Params+grads (GB): 372.697
# Activations  (GB): 2.678
# Optimizer    (GB): 186.349
# Persistent   (GB): 561.724
# Temporaries  (GB): 0.227
# Batch=1/4 Epoch=1/1 Loss=24.501052856445312
# Batch=2/4 Epoch=1/1 Loss=24.430049896240234
# Batch=3/4 Epoch=1/1 Loss=23.843454360961914
# Batch=4/4 Epoch=1/1 Loss=24.420440673828125
# NNTile training time: 2139.3117303848267 seconds
# NNTile training throughput tokens/sec: 0.47865861971214424
# NNTile performance (model flops): 0.2803519863656865 Tflops/s
# NNTile loss on the last batch: 24.420440673828125
