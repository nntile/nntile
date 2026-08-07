python mlm_data_preparation.py --seq-len=256 --batch-size=1 --dataset-select=4

STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA=4 python bert_training.py --pretrained=local --config-path="bert_config_100b.json"  --optimizer="sgd" --lr=1e-5 --dtype=bf16 --nepochs=1 --batch-size=1 --minibatch-size=1 --n-masks-per-seq=2 --seq-len=256 --dataset-file="tinystories/train.bin" --save-checkpoint-path="" --hidden-size-tile=8650 --intermediate-size-tile=34600 --n-head-tile=1 --restrict=cuda

# Number of parameters: 102595604522
# Converting PyTorch model to NNTile requires 3859.641193628311 seconds
# From PyTorch loader to NNTile batches in 0.0043430328369140625 seconds
# Params+grads (GB): 382.198
# Activations  (GB): 2.009
# Optimizer    (GB): 191.099
# Persistent   (GB): 575.307
# Temporaries  (GB): 0.910
# Batch=1/4 Epoch=1/1 Loss=19.95833396911621
# Batch=2/4 Epoch=1/1 Loss=25.0244140625
# Batch=3/4 Epoch=1/1 Loss=20.31770896911621
# Batch=4/4 Epoch=1/1 Loss=21.6953125
# NNTile training time: 850.6150510311127 seconds
# NNTile training throughput tokens/sec: 0.6019174000969714
# NNTile loss on the last batch: 21.6953125
