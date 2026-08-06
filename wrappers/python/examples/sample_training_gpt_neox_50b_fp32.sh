python causal_lm_data_preparation.py --hf-tokenizer="EleutherAI/gpt-neox-20b" --seq-len=256 --batch-size=1 --dataset-select=7

STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA=4 python gpt_neox_training.py --restrict="cuda" --pretrained=local --config-path="gpt_neox_config_50b.json" --save-checkpoint-path="" --optimizer="sgd" --seq-len=256 --lr=1e-5 --dtype=fp32 --nepochs=1 --batch-size=1 --minibatch-size=1 --dataset-file="tinystories/train.bin" --hidden-size-tile=6048 --intermediate-size-tile=24192 --n-head-tile=1

# Number of parameters: 49976498880
# From PyTorch loader to NNTile batches in 0.00035858154296875 seconds
# Params+grads (GB): 372.354
# Activations  (GB): 1.942 
# Optimizer    (GB): 186.177
# Persistent   (GB): 560.473
# Temporaries  (GB): 1.510 
# Batch=1/4 Epoch=1/1 Loss=21.447189331054688
# Batch=2/4 Epoch=1/1 Loss=22.259349822998047
# Batch=3/4 Epoch=1/1 Loss=21.792089462280273
# Batch=4/4 Epoch=1/1 Loss=21.226770401000977
# NNTile training time: 503.60977578163147 seconds 
# NNTile training throughput tokens/sec: 2.033320338968188
# NNTile performance (model flops): 0.39410112813310294 Tflops/s
# NNTile loss on the last batch: 21.226770401000977
