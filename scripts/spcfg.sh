#!/bin/bash -l

PORT_ID=$(expr $RANDOM + 1000)
NUM_GPU=4
export OMP_NUM_THREADS=4

python -m torch.distributed.launch --use_env --nproc_per_node $NUM_GPU --master_port $PORT_ID src/run.py \
    --train_on "${DATASET}" \
    --model_type spcfg \
    --max_dev_examples 1024 \
    --max_length 225 \
    --max_eval_length 225 \
    --min_length 6 \
    --use_product_length 1 \
    --epochs 10 \
    --patience 5 \
    --eval_every 1024 \
    --eval_before_training \
    --eval_on "${DATASET}" \
    --eval_on_train_datasets "${DATASET}" \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --criterion loss \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 4 \
    --save \
    --preterminals 64 \
    --nonterminals 32 \
    --output_dir "output/${DATASET}/spcfg";

# Finish the script
exit 0
