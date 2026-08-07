python mlm_data_preparation.py --seq-len=256 --batch-size=1 --dataset-select=4

STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA=4 python bert_training.py --pretrained=local --config-path="bert_config.json"  --optimizer="sgd" --lr=1e-5 --dtype=fp32 --nepochs=1 --batch-size=1 --minibatch-size=1 --n-masks-per-seq=2 --seq-len=256 --dataset-file="tinystories/train.bin" --save-checkpoint-path="" --hidden-size-tile=6080 --intermediate-size-tile=24320 --n-head-tile=1 --restrict=cuda

# Number of parameters: 51800171322
# Converting PyTorch model to NNTile requires 1094.7564985752106 seconds
# From PyTorch loader to NNTile batches in 0.002259969711303711 seconds
# Params+grads (GB): 385.941
# Activations  (GB): 2.841
# Optimizer    (GB): 192.971
# Persistent   (GB): 581.753
# Temporaries  (GB): 1.325
# Batch=1/4 Epoch=1/1 Loss=18.473529815673828
# Batch=2/4 Epoch=1/1 Loss=26.239103317260742
# Batch=3/4 Epoch=1/1 Loss=18.23354721069336
# Batch=4/4 Epoch=1/1 Loss=22.817548751831055
# NNTile training time: 2887.3574364185333 seconds
# NNTile training throughput tokens/sec: 0.1773247723132896
# NNTile loss on the last batch: 22.817548751831055
