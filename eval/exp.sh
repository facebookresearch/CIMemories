
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

MODEL_NAME="gpt-4o"
MODEL_HOST="localhost"
MODEL_PORT=8000
STRONG_MODEL_NAME="deepseek-ai/DeepSeek-R1-0528"
STRONG_MODEL_HOST="localhost"
STRONG_MODEL_PORT=8000

GREEDY="false"
DATA_FILE="../data/data_openai_gpt-oss-120b_gold_labelled_personas_combined.json"
NUM_PROFILES=1
RESULTS_DIR="./results"
PROMPTS_FILE="./prompts.yaml"
PRIVACY_PROMPTS_LEVEL=1
NUM_TRIALS=5
COMBO_STYLE="full"

export AZURE_API_KEY_gpt_4o="KEY_HERE"
export AZURE_API_BASE_gpt_4o="EDNPOINT_HERE"
export AZURE_API_VERSION_gpt_4o="2024-06-01"

python eval.py --model_name $MODEL_NAME \
                --model_host $MODEL_HOST \
                --model_port $MODEL_PORT \
                --strong_model_name $STRONG_MODEL_NAME \
                --strong_model_host $STRONG_MODEL_HOST \
                --strong_model_port $STRONG_MODEL_PORT \
                $(if [ "$GREEDY" == "true" ]; then echo "--greedy"; fi) \
                --data_file $DATA_FILE \
                --num_profiles $NUM_PROFILES \
                --model_port $MODEL_PORT \
                --results_dir $RESULTS_DIR \
                --prompts_file $PROMPTS_FILE \
                --privacy_prompts_level $PRIVACY_PROMPTS_LEVEL \
                --num_trials $NUM_TRIALS \
                --combo_style $COMBO_STYLE