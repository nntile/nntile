python mlm_data_preparation.py --seq-len=256 --batch-size=1 --dataset-select=7 --hf-tokenizer="FacebookAI/roberta-base"

STARPU_SILENT=1 STARPU_NCUDA=4 STARPU_NCPU=1 python roberta_training.py --pretrained=local --config-path="roberta_config_100b.json" --save-checkpoint-path="" --optimizer="sgd" --lr=1e-5 --dtype=bf16 --nepochs=1  --batch-size=1 --minibatch-size=1 --seq-len=256 --dataset-file="tinystories/train.bin" --restrict="cuda" --hidden-size-tile=8580 --intermediate-size-tile=34320 --n-head-tile=1

# Number of parameters: 104372575665
# Converting PyTorch model to NNTile requires 3289.6399524211884 seconds
# From PyTorch loader to NNTile batches in 0.0010771751403808594 seconds
# Params+grads (GB): 388.818
# Activations  (GB): 2.012
# Optimizer    (GB): 194.409
# Persistent   (GB): 585.239
# Temporaries  (GB): 0.903
# Batch=1/4 Epoch=1/1 Loss=22.893230438232422
# Batch=2/4 Epoch=1/1 Loss=20.63802146911621
# Batch=3/4 Epoch=1/1 Loss=27.838542938232422
# Batch=4/4 Epoch=1/1 Loss=22.683269500732422
# NNTile training time: 1993.0800869464874 seconds
# NNTile training throughput tokens/sec: 0.5137776483276327
# NNTile loss on the last batch: 22.683269500732422

