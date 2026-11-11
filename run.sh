#!/bin/bash
# Paired baseline / LLM-JEPA runs, one per seed (port of upstream run.sh).
# Usage: ./run.sh [dataset] [learning_rate] [lbd] [predictors] [extra finetune.py flags...]
set -euo pipefail

model=${MODEL:-mlx-community/Llama-3.2-1B-Instruct-bf16}
dataset=${1:-synth}
learning_rate=${2:-2e-5}
lbd=${3:-1.0}
predictors=${4:-1}
extra=("${@:5}")
epochs=4
# Token before the closing turn marker. Upstream: -2 llama/gemma, -1 OLMo-2/DeepSeek, -3 Qwen.
last_token=${LAST_TOKEN:--2}

evaluate() {
  uv run python evaluate.py --model_name=./fine-tuned --original_model_name="${model}" \
    --input_file="datasets/${dataset}_test.jsonl" --output_file=eval.jsonl \
    --spider_path=spider_data/database | tee -a output.txt
}

for seed in 82 23 37 84 4; do
  echo "Success Rate: regular ${model} lr=${learning_rate} e=${epochs} s=${seed} dataset=${dataset}" >> output.txt
  uv run python finetune.py --train_file "datasets/${dataset}_train.jsonl" --model_name="${model}" \
    --output_dir=./fine-tuned --num_epochs=${epochs} --finetune_seed=${seed} \
    --learning_rate=${learning_rate} --regular ${extra[@]+"${extra[@]}"}
  evaluate

  echo "Success Rate: jepa ${model} lr=${learning_rate} e=${epochs} lt=${last_token} p=${predictors} s=${seed} lbd=${lbd} dataset=${dataset}" >> output.txt
  uv run python finetune.py --train_file "datasets/${dataset}_train.jsonl" --model_name="${model}" \
    --output_dir=./fine-tuned --num_epochs=${epochs} --finetune_seed=${seed} \
    --learning_rate=${learning_rate} --last_token=${last_token} --lbd=${lbd} \
    --predictors=${predictors} ${extra[@]+"${extra[@]}"}
  evaluate
done
