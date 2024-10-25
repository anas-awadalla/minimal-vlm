import os
import argparse
import glob
import random
import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DataParallel as DDP
from torch import autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, SiglipVisionModel, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
import webdataset as wds
import wandb
import numpy as np
import time

from vlm import VLM
from data import create_dataloader, preprocess_image

os.system("export HF_HOME=/mmfs1/gscratch/xlab/anasa2/hf_cache")

def parse_args():
    parser = argparse.ArgumentParser(description="Train VLM model on WebDataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of total steps to use for warmup")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--lm_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Language model name")
    parser.add_argument("--vision_encoder_name", type=str, default="google/siglip-so400m-patch14-384", help="Vision encoder name")
    parser.add_argument("--use_perceiver", action="store_true", help="Use Perceiver Resampler")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for saved models")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--wandb_project", type=str, default="VLM_training", help="Weights & Biases project name")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model weights")
    parser.add_argument("--base_seed", type=int, default=42, help="Base seed for data shuffling")
    parser.add_argument("--caption_key", type=str, default="caption", help="Key for caption in the dataset")
    parser.add_argument("--dataset_type", type=str, choices=["caption", "cauldron"], required=True, help="Type of dataset to use for training")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing dataset files")
    parser.add_argument("--total_samples", type=int, required=True, help="Number of samples to process in training")
    return parser.parse_args()

def calculate_num_image_tokens(vision_encoder):
    config = vision_encoder.config
    image_size = config.image_size
    patch_size = config.patch_size
    return (image_size // patch_size) ** 2

def save_checkpoint(model, optimizer, scheduler, epoch, step, output_dir, filename):
    checkpoint_path = os.path.join(output_dir, filename)
    torch.save({
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
        'args': args
    }, checkpoint_path)
    return checkpoint_path

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(args):
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    
    # Set seeds for reproducibility
    torch.manual_seed(args.base_seed)
    torch.cuda.manual_seed_all(args.base_seed)
    np.random.seed(args.base_seed)
    random.seed(args.base_seed)

    if args.local_rank == -1:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
    else:
        local_rank = args.local_rank

    if local_rank == -1:
        raise ValueError("Local rank not set")
        
    print("Starting training on rank ", local_rank)
        
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    print("World size: ", world_size, "Rank: ", rank, "Local rank: ", local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.set_float32_matmul_precision('high')
    
    # Initialize wandb only on the main process
    if rank == 0:
        wandb.init(project=args.wandb_project, config=args, name=args.output_dir)
        wandb.config.update(args)

    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    tokenizer.pad_token = tokenizer.eos_token
    lm = AutoModelForCausalLM.from_pretrained(args.lm_name)
    image_processor = AutoImageProcessor.from_pretrained(args.vision_encoder_name)

    model = VLM(lm, args.vision_encoder_name, use_perceiver=args.use_perceiver).to(device)
    
    num_image_tokens = calculate_num_image_tokens(model.vision_encoder) if not args.use_perceiver else 64
    
    if args.gradient_checkpointing:
        model.lm.gradient_checkpointing_enable()
            
    if args.pretrained is not None:
        print(f"Loading pretrained model from {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model'])
                
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    total_steps = args.total_samples // (args.batch_size * world_size)

    print(f"Total steps: {total_steps}")

    # Warmup steps for the training phase
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"Warmup steps: {warmup_steps}")

    global_step = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    for epoch in range(args.epochs):
        dataloader = create_dataloader(args, tokenizer, image_processor, num_image_tokens, epoch, world_size, rank, base_seed=args.base_seed, total_samples=args.total_samples)
        
        if rank == 0:
            progress_bar = tqdm(enumerate(dataloader, start=global_step+1), total=total_steps, initial=global_step, desc=f"Epoch {epoch + 1}/{args.epochs}")
        else:
            progress_bar = enumerate(dataloader, start=global_step+1)
        
        model.train()
        total_loss = 0
        steps_this_epoch = 0
        samples_this_epoch = 0
        epoch_start_time = time.time()
        
        consecutive_oom_count = 0
        for step, (pixel_values, (input_ids, attention_mask, labels)) in progress_bar:
            try:
                pixel_values = pixel_values.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs, loss = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=labels
                    )
                    
                next_token_loss_item = loss.item()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += next_token_loss_item
                
                steps_this_epoch += 1
                samples_this_epoch += args.batch_size * world_size
                global_step += 1

                if rank == 0:
                    elapsed_time = time.time() - epoch_start_time
                    samples_per_second = samples_this_epoch / elapsed_time
                    current_lr = get_lr(optimizer)
                    progress_bar.set_postfix({
                        'next_token_loss': f'{next_token_loss_item:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'samples/s': f'{samples_per_second:.2f}'
                    })
                    log_dict = {
                        "step": global_step,
                        "loss": next_token_loss_item,
                        "learning_rate": current_lr,
                        "samples_per_second": samples_per_second
                    }
                    wandb.log(log_dict)

                consecutive_oom_count = 0  # Reset the counter on successful execution

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                consecutive_oom_count += 1
                if rank == 0:
                    print(f"OOM error encountered. Skipping batch at step {global_step}. Consecutive OOMs: {consecutive_oom_count}")
                
                if consecutive_oom_count > 5:
                    raise RuntimeError("Encountered more than 5 consecutive OOM errors. Stopping training.")
                
                continue

            if global_step >= total_steps:
                break

    # Save final checkpoint at the end of training
    if rank == 0:
        print("Training completed. Saving final checkpoint...")
        os.makedirs(args.output_dir, exist_ok=True)
        final_checkpoint_path = save_checkpoint(
            model, optimizer, scheduler, args.epochs, global_step, 
            args.output_dir, "VLM_model_final.pt"
        )
        print(f"Final full state checkpoint saved at {final_checkpoint_path}")

        print("Training completed!")
        wandb.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    train(args)

# Example command:
# torchrun --nproc_per_node=4 /mmfs1/gscratch/xlab/anasa2/kale_experiments/train.py --batch_size 24 --learning_rate 2e-5 --warmup_ratio 0.1 --epochs 1 --output_dir model-output --dataset_type caption --data_dir /mmfs1/gscratch/xlab/anasa2/kale-caption-wds --total_samples 1000000 --gradient_checkpointing --lm_name meta-llama/Llama-3.2-1B-Instruct --vision_encoder_name google/siglip-so400m-patch14-384 --caption_key caption
