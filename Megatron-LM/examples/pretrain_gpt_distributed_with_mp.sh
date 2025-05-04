#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
# export LD_PRELOAD=$LD_PRELOAD:/workspace/ncclprobe/build/libncclprobe.so
# export CONTROL_PLANE_WHL_PATH=/workspace/ncclprobe/dist/control_plane-1.0-py3-none-any.whl
# export NCCLPROBE_LOG_PATH=/workspace/trainlogs/
# export GLOBAL_CONTROLLER_LOG_PATH=/workspace/trainlogs/
# export LOCAL_CONTROLLER_LOG_PATH=/workspace/trainlogs/

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=./checkpoints
VOCAB_FILE=./dataset/gpt2-vocab.json
MERGE_FILE=./dataset/gpt2-merges.txt

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 1\
    --pipeline-model-parallel-size 1\
    --sequence-parallel \
    --num-layers 32 \
    --hidden-size 1024 \
    --num-attention-heads 32 \
    --seq-length 32 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 64 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --mock-data
    --failslow-aware
"

DATA_ARGS="
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

redis-server --bind $MASTER_ADDR --save \"\" &

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH

