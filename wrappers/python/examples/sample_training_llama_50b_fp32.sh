python causal_lm_data_preparation.py --seq-len=256 --batch-size=1 --dataset-select=6

STARPU_SILENT=1 STARPU_NCUDA=4 STARPU_NCPU=1 python llama_training.py --restrict="cuda" --pretrained=local --config-path="llama_config_50b.json" --save-checkpoint-path="" --optimizer="sgd" --seq-len=256 --lr=1e-5 --dtype=fp32 --nepochs=1 --batch-size=1 --minibatch-size=1 --dataset-file="tinystories/train.bin" --hidden-size-tile=5396 --intermediate-size-tile=21580 --n-head-tile=1

# Number of parameters: 50034032280
# Converting PyTorch model to NNTile requires 924.7585852146149 seconds
# From PyTorch loader to NNTile batches in 0.0003676414489746094 seconds
# Params+grads (GB): 372.783
# Activations  (GB): 3.766
# Optimizer    (GB): 186.391
# Persistent   (GB): 562.940
# Temporaries  (GB): 0.319
# Batch=1/4 Epoch=1/1 Loss=19.63129425048828
# Batch=2/4 Epoch=1/1 Loss=20.439725875854492
# Batch=3/4 Epoch=1/1 Loss=20.453474044799805
# Batch=4/4 Epoch=1/1 Loss=19.803747177124023
# NNTile training time: 1974.6944510936737 seconds
# NNTile training throughput tokens/sec: 0.5185612383894953
# NNTile performance (model flops): 0.15038725377285858 Tflops/s
# NNTile loss on the last batch: 19.803747177124023
