#!/bin/bash

run_training() {
    local model_name=$1
    local caption_key=$2
    local output_dir="${model_name}-caption-model-2m-qwen-1.5B-so400m-4gpu-instruct-linear-proj"
    local data_dir=$3

    torchrun --nproc_per_node=4 /mmfs1/gscratch/xlab/anasa2/kale_experiments/train.py \
            --batch_size 20 \
            --learning_rate 5e-5 \
            --epochs 1 \
            --base_seed 42 \
            --output_dir "${output_dir}" \
            --dataset_type caption \
            --data_dir "${data_dir}" \
            --total_samples 2000000 \
            --gradient_checkpointing \
            --lm_name Qwen/Qwen2.5-1.5B-Instruct \
            --vision_encoder_name google/siglip-so400m-patch14-384 \
            --caption_key "${caption_key}"

    # Immediately run cauldron training after caption training
    local cauldron_output_dir="${model_name}-cauldron-model-2m-qwen-1.5B-so400m-4gpu-instruct-linear-proj"
    torchrun --nproc_per_node=4 /mmfs1/gscratch/xlab/anasa2/kale_experiments/train.py \
            --batch_size 20 \
            --learning_rate 3e-5 \
            --epochs 1 \
            --base_seed 42 \
            --output_dir "${cauldron_output_dir}" \
            --dataset_type cauldron \
            --data_dir /mmfs1/gscratch/xlab/anasa2/cauldron_webdataset_shards_shuffled_2M \
            --total_samples 1000000 \
            --gradient_checkpointing \
            --lm_name Qwen/Qwen2.5-1.5B-Instruct \
            --vision_encoder_name google/siglip-so400m-patch14-384 \
            --pretrained "${output_dir}/VLM_model_final.pt"
}

run_training "cogvlm" "cogvlm_caption" "/mmfs1/gscratch/xlab/anasa2/kale-caption-wds"

echo "All training runs completed."
