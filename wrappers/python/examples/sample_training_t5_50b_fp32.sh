python causal_lm_data_preparation.py --hf-tokenizer="google/flan-t5-small" --seq-len=256 --batch-size=1 --dataset-select=12

STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA=4 python t5_lmhead_training.py --restrict="cuda" --pretrained=local --config-path="t5_config_50b.json" --save-checkpoint-path="" --optimizer="sgd" --lr=1e-5 --dtype=fp32 --nepochs=1 --dataset-file="tinystories/train.bin" --num-heads-tile=1 --d-model-tile=3640 --d-ff-tile=14560 --batch-size=1 --minibatch-size=1

# Number of parameters: 50035695440
# Converting PyTorch model to NNTile requires 1019.0103924274445 seconds
# train_labels:  (4, 512)
# train_tokens:  30718
# From PyTorch loader to NNTile batches in 0.0005252361297607422 seconds
# Params+grads (GB): 372.795
# Activations  (GB): 7.204
# Optimizer    (GB): 186.397
# Persistent   (GB): 566.397
# Temporaries  (GB): 4.402
# Batch=1/4 Epoch=1/1 Loss=29483.177734375
# Batch=2/4 Epoch=1/1 Loss=19706.7578125
# Batch=3/4 Epoch=1/1 Loss=17952.62890625
# Batch=4/4 Epoch=1/1 Loss=15672.515625
# NNTile training time: 1176.3728098869324 seconds
# NNTile training throughput samples/sec: 0.003400282602914345
# NNTile performance (model flops): 0.3443673560246065 Tflops/s
# NNTile loss on the last batch: 15672.515625
