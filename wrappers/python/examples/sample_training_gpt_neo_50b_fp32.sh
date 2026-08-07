python causal_lm_data_preparation.py --hf-tokenizer="EleutherAI/gpt-neo-1.3B" --seq-len=256 --batch-size=1 --dataset-select=7

STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA=4 python gpt_neo_training.py --pretrained=local --config-path="gpt_neo_config_50b.json" --save-checkpoint-path="" --optimizer="sgd" --seq-len=256 --lr=1e-5 --dtype=fp32 --nepochs=1 --batch-size=1 --minibatch-size=1 --dataset-file="tinystories/train.bin" --n-head-tile=1 --hidden-size-tile=6240 --intermediate-size-tile=24960 --restrict=cuda

# Number of parameters: 53125737600
# Converting PyTorch model to NNTile requires 1117.5141739845276 seconds
# From PyTorch loader to NNTile batches in 0.0004432201385498047 seconds
# Params+grads (GB): 395.818
# Activations  (GB): 2.238
# Optimizer    (GB): 197.909
# Persistent   (GB): 595.965
# Temporaries  (GB): 1.358
# Batch=1/4 Epoch=1/1 Loss=22.948904037475586
# Batch=2/4 Epoch=1/1 Loss=22.301780700683594
# Batch=3/4 Epoch=1/1 Loss=22.12322998046875
# Batch=4/4 Epoch=1/1 Loss=23.06597328186035
# NNTile training time: 2214.5810983181 seconds
# NNTile training throughput samples/sec: 0.0018062106657723512
# NNTile performance (model flops): 0.09512133097278964 Tflops/s
# NNTile loss on the last batch: 23.06597328186035
