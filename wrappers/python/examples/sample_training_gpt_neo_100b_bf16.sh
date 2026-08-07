python causal_lm_data_preparation.py --hf-tokenizer="EleutherAI/gpt-neo-1.3B" --seq-len=256 --batch-size=1 --dataset-select=7

STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA=4 python gpt_neo_training.py --pretrained=local --config-path="gpt_neo_config_100b.json" --save-checkpoint-path="" --optimizer="sgd" --seq-len=256 --lr=1e-5 --dtype=bf16 --nepochs=1 --batch-size=1 --minibatch-size=1 --dataset-file="tinystories/train.bin" --n-head-tile=1 --hidden-size-tile=8910 --intermediate-size-tile=35640 --restrict=cuda

# Number of parameters: 104405063400
# Converting PyTorch model to NNTile requires 4905.037987232208 seconds
# train_labels:  (4, 256)
# train_labels:  44193
# From PyTorch loader to NNTile batches in 0.0005197525024414062 seconds
# Params+grads (GB): 388.939
# Activations  (GB): 1.577
# Optimizer    (GB): 194.470
# Persistent   (GB): 584.986
# Temporaries  (GB): 0.937
# Batch=1/4 Epoch=1/1 Loss=26.238994598388672
# Batch=2/4 Epoch=1/1 Loss=26.365985870361328
# Batch=3/4 Epoch=1/1 Loss=26.628328323364258
# Batch=4/4 Epoch=1/1 Loss=26.631755828857422
# NNTile training time: 364.5408823490143 seconds
# NNTile training throughput samples/sec: 0.010972706200261974
# NNTile performance (model flops): 1.1458813509231347 Tflops/s
# NNTile loss on the last batch: 26.631755828857422

