python causal_lm_data_preparation.py --hf-tokenizer="openai-community/gpt2" --seq-len=256 --batch-size=1 --dataset-select=25

STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA=4 python gpt2_lmhead_training.py --restrict="cuda" --pretrained=local --config-path="gpt2_config_50b.json" --save-checkpoint-path="" --optimizer="sgd" --lr=1e-5 --dtype=fp32 --nepochs=1 --batch-size=1 --minibatch-size=1 --dataset-file="tinystories/train.bin" --hidden-size-tile=6245 --intermediate-size-tile=24980 --n-head-tile=1

# Number of parameters: 53142014850
# Converting PyTorch model to NNTile requires 228.21810698509216 seconds
# From PyTorch loader to NNTile batches in 0.00037741661071777344 seconds
# Params+grads (GB): 395.939
# Activations  (GB): 12.771
# Optimizer    (GB): 197.969
# Persistent   (GB): 606.680
# Temporaries  (GB): 1.508
# Batch=1/4 Epoch=1/1 Loss=22.28024673461914
# Batch=2/4 Epoch=1/1 Loss=22.549680709838867
# Batch=3/4 Epoch=1/1 Loss=22.174861907958984
# Batch=4/4 Epoch=1/1 Loss=22.363784790039062
# NNTile training time: 611.9939742088318 seconds
# NNTile training throughput tokens/sec: 6.6928763560052875
# NNTile performance (model flops): 2.00710843959789 Tflops/s
# NNTile loss on the last batch: 22.363784790039062
