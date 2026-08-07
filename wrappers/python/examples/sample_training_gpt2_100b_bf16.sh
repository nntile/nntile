python causal_lm_data_preparation.py --hf-tokenizer="openai-community/gpt2" --seq-len=256 --batch-size=1 --dataset-select=25

STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA=4 python gpt2_lmhead_training.py --restrict="cuda" --pretrained=local --config-path="gpt2_config_100b.json" --save-checkpoint-path="" --optimizer="sgd" --lr=1e-5 --dtype=bf16 --nepochs=1 --batch-size=1 --minibatch-size=1 --dataset-file="tinystories/train.bin" --hidden-size-tile=8910 --intermediate-size-tile=35640 --n-head-tile=1

# Number of parameters: 104314092300
# Converting PyTorch model to NNTile requires 617.7376816272736 seconds
# From PyTorch loader to NNTile batches in 0.000377655029296875 seconds
# Params+grads (GB): 388.600
# Activations  (GB): 9.029
# Optimizer    (GB): 194.300
# Persistent   (GB): 591.929
# Temporaries  (GB): 1.059
# Batch=1/4 Epoch=1/1 Loss=25.420425415039062
# Batch=2/4 Epoch=1/1 Loss=25.97182273864746
# Batch=3/4 Epoch=1/1 Loss=25.685686111450195
# Batch=4/4 Epoch=1/1 Loss=25.528518676757812
# NNTile training time: 2115.7993862628937 seconds
# NNTile training throughput tokens/sec: 1.9359113281693054
# NNTile performance (model flops): 1.159275325932075 Tflops/s
# NNTile loss on the last batch: 25.528518676757812
