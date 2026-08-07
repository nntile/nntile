python causal_lm_data_preparation.py --hf-tokenizer="EleutherAI/gpt-neox-20b" --seq-len=256 --batch-size=1 --dataset-select=7

STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA=4 python gpt_neox_training.py --restrict="cuda" --pretrained=local --config-path="gpt_neox_config_100b.json" --save-checkpoint-path="" --optimizer="sgd" --seq-len=256 --lr=1e-5 --dtype=bf16 --nepochs=1 --batch-size=1 --minibatch-size=1 --dataset-file="tinystories/train.bin" --hidden-size-tile=8720 --intermediate-size-tile=34880 --n-head-tile=1

# Number of parameters: 100015871200
# From PyTorch loader to NNTile batches in 0.0030946731567382812 seconds
# Params+grads (GB): 372.588
# Activations  (GB): 1.379
# Optimizer    (GB): 186.294
# Persistent   (GB): 560.261
# Temporaries  (GB): 1.088
# Batch=1/4 Epoch=1/1 Loss=24.949745178222656
# Batch=2/4 Epoch=1/1 Loss=25.545373916625977
# Batch=3/4 Epoch=1/1 Loss=25.598873138427734
# Batch=4/4 Epoch=1/1 Loss=25.350534439086914
# NNTile training time: 1171.0974643230438 seconds
# NNTile training throughput tokens/sec: 0.8743934908884173
# NNTile performance (model flops): 0.3421421425548174 Tflops/s
# NNTile loss on the last batch: 25.350534439086914
