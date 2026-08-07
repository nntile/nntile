python causal_lm_data_preparation.py --hf-tokenizer="google/flan-t5-small" --seq-len=256 --batch-size=1 --dataset-select=12

STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA=4 python t5_lmhead_training.py --restrict="cuda" --pretrained=local --config-path="t5_config_100b.json" --save-checkpoint-path="" --optimizer="sgd" --lr=1e-5 --dtype=bf16 --nepochs=1 --seq-len=256 --dataset-file="tinystories/train.bin" --num-heads-tile=1 --d-model-tile=5220 --d-ff-tile=20880 --batch-size=1 --minibatch-size=1


# Number of parameters: 101445846040
# Converting PyTorch model to NNTile requires 2467.135982275009 seconds
# train_labels:  (8, 256)
# train_tokens:  30718
# From PyTorch loader to NNTile batches in 0.0007169246673583984 seconds
# Params+grads (GB): 377.915
# Activations  (GB): 2.569
# Optimizer    (GB): 188.958
# Persistent   (GB): 569.442
# Temporaries  (GB): 1.555
# Batch=1/8 Epoch=1/1 Loss=41791.06640625
# Batch=2/8 Epoch=1/1 Loss=28220.837890625
# Batch=3/8 Epoch=1/1 Loss=27288.853515625
# Batch=4/8 Epoch=1/1 Loss=26355.12109375
# Batch=5/8 Epoch=1/1 Loss=25810.431640625
# Batch=6/8 Epoch=1/1 Loss=24390.966796875
# Batch=7/8 Epoch=1/1 Loss=26553.115234375
# Batch=8/8 Epoch=1/1 Loss=25920.9296875
# NNTile training time: 1189.0885150432587 seconds
# NNTile training throughput samples/sec: 0.006727842291630377
# NNTile performance (model flops): 0.6931174779785142 Tflops/s
# NNTile loss on the last batch: 25920.9296875
