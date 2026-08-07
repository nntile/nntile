python mlm_data_preparation.py --seq-len=256 --batch-size=1 --dataset-select=7 --hf-tokenizer="FacebookAI/roberta-base"

STARPU_SILENT=1 STARPU_NCUDA=4 STARPU_NCPU=1 python roberta_training.py --pretrained=local --config-path="roberta_config_50b.json" --save-checkpoint-path="" --optimizer="sgd" --lr=1e-5 --dtype=fp32 --nepochs=1  --batch-size=1 --minibatch-size=1 --seq-len=256 --dataset-file="tinystories/train.bin" --restrict="cuda" --hidden-size-tile=6010 --intermediate-size-tile=24040 --n-head-tile=1

# Number of parameters: 53030066565
# Converting PyTorch model to NNTile requires 1061.8977558612823 seconds
# From PyTorch loader to NNTile batches in 0.0010645389556884766 seconds
# Params+grads (GB): 395.105
# Activations  (GB): 2.847
# Optimizer    (GB): 197.552
# Persistent   (GB): 595.504
# Temporaries  (GB): 1.266
# Batch=1/4 Epoch=1/1 Loss=18.521291732788086
# Batch=2/4 Epoch=1/1 Loss=17.874698638916016
# Batch=3/4 Epoch=1/1 Loss=19.87941551208496
# Batch=4/4 Epoch=1/1 Loss=21.171403884887695
# NNTile training time: 1955.3893892765045 seconds
# NNTile training throughput tokens/sec: 0.5236808615284962
# NNTile loss on the last batch: 21.171403884887695
