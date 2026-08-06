python causal_lm_data_preparation.py --hf-tokenizer="EleutherAI/gpt-neox-20b" --seq-len=256 --batch-size=1 --dataset-select=7

STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA=4 python gpt_neox_training.py --restrict="cuda" --pretrained=local --config-path="gpt_neox_config_50b.json" --save-checkpoint-path="" --optimizer="sgd" --seq-len=256 --lr=1e-5 --dtype=fp32 --nepochs=1 --batch-size=1 --minibatch-size=1 --dataset-file="tinystories/train.bin" --hidden-size-tile=6048 --intermediate-size-tile=24192 --n-head-tile=1
