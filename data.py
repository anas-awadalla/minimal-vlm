import os
import glob
import random
import torch
from torch.utils.data import IterableDataset, DistributedSampler
import webdataset as wds
import random
import tarfile
import json
from PIL import Image
import io
import math

def get_random_caption_prompt():
    prompts = [
        "Describe this image in detail.",
        "What do you see in this picture?",
        "Can you explain what's in this image?",
        "Describe the scene captured in this picture.",
        "If you had to summarize this image in a few sentences, what would you say?",
        "Can you provide a detailed description of what you observe here?",
        "How would you describe this image to someone who can't see it?",
        "Describe the image as if you're seeing it for the first time.",
    ]
    return random.choice(prompts)

def find_subsequence(seq, subseq):
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i:i+m] == subseq:
            return i
    return -1

def preprocess_cauldron_sample(sample, tokenizer, max_length, num_image_tokens):
    texts = sample['texts']
    formatted_text = ""
    
    if "qwen" in tokenizer.name_or_path.lower():
        start_token, end_token = "<|im_start|>", None
        eot_token = "<|im_end|>"
    else:
        start_token, end_token = "<|start_header_id|>", "<|end_header_id|>"
        eot_token = "<|eot_id|>"
    
    for qa_pair in texts:
        if "qwen" in tokenizer.name_or_path.lower():
            formatted_text += f"{start_token}user\n{qa_pair['user']}{eot_token}{start_token}assistant\n{qa_pair['assistant'].split('Answer:')[-1].strip()}{eot_token}\n"
        else:
            formatted_text += f"{start_token}user{end_token}\n\n{qa_pair['user']}{eot_token}{start_token}assistant{end_token}\n\n{qa_pair['assistant'].split('Answer:')[-1].strip()}{eot_token}\n\n"

    formatted_text = formatted_text.strip() # Remove trailing newlines

    encoding = tokenizer(formatted_text, padding="max_length", truncation=True,
                         max_length=max_length, return_tensors="pt")
    input_ids = encoding.input_ids.squeeze()
    attention_mask = encoding.attention_mask.squeeze()
    labels = input_ids.clone()

    if "qwen" in tokenizer.name_or_path.lower():
        user_tokens = tokenizer.encode(f"{start_token}user\n", add_special_tokens=False)
        assistant_tokens = tokenizer.encode(f"{start_token}assistant\n", add_special_tokens=False)
    else:
        user_tokens = tokenizer.encode(f"{start_token}user{end_token}\n\n", add_special_tokens=False)
        assistant_tokens = tokenizer.encode(f"{start_token}assistant{end_token}\n\n", add_special_tokens=False)

    mask_starts = []
    start = 0
    while True:
        idx_user = find_subsequence(input_ids[start:].tolist(), user_tokens)
        if idx_user == -1:
            break
        mask_starts.append(start + idx_user)
        start = start + idx_user + len(user_tokens)
    
    for i, start in enumerate(mask_starts):
        end = find_subsequence(input_ids[start:].tolist(), assistant_tokens)
        if end != -1:
            end = start + end + len(assistant_tokens)  # Include assistant tokens in the mask
            labels[start:end] = -100
            if i < len(mask_starts) - 1:
                next_start = mask_starts[i+1]
        else:
            labels[start:] = -100
            
    # mask loss on padding using attention mask
    labels[attention_mask == 0] = -100
    
    # labelled_tokens = tokenizer.decode(input_ids[labels != -100])
    # unlabelled_tokens = tokenizer.decode(input_ids[labels == -100])
    # print(f"Labelled tokens: {repr(labelled_tokens)} -- Unlabelled tokens: {repr(unlabelled_tokens)} \n ---------------------------------------- \n")

    labels = torch.cat([torch.full((num_image_tokens,), -100), labels])
    
    return input_ids, attention_mask, labels

def preprocess_caption_sample(sample, tokenizer, max_length, num_image_tokens, caption_key):
    caption = sample[caption_key]
    prompt = get_random_caption_prompt()
    
    if "qwen" in tokenizer.name_or_path.lower():
        start_token, end_token = "<|im_start|>", None
        eot_token = "<|im_end|>"
    else:
        start_token, end_token = "<|start_header_id|>", "<|end_header_id|>"
        eot_token = "<|eot_id|>"
            
    if "qwen" in tokenizer.name_or_path.lower():
        formatted_text = f"{start_token}user\n{prompt}{eot_token}{start_token}assistant\n{caption}{eot_token}"
    else:
        formatted_text = f"{start_token}user{end_token}\n\n{prompt}{eot_token}{start_token}assistant{end_token}\n\n{caption}{eot_token}"

    encoding = tokenizer(formatted_text, padding="max_length", truncation=True,
                         max_length=max_length, return_tensors="pt")
    
    input_ids = encoding.input_ids.squeeze()
    attention_mask = encoding.attention_mask.squeeze()
    labels = input_ids.clone()

    if "qwen" in tokenizer.name_or_path.lower():
        user_tokens = tokenizer.encode(f"{start_token}user\n", add_special_tokens=False)
        assistant_tokens = tokenizer.encode(f"{start_token}assistant\n", add_special_tokens=False)
    else:
        user_tokens = tokenizer.encode(f"{start_token}user{end_token}\n\n", add_special_tokens=False)
        assistant_tokens = tokenizer.encode(f"{start_token}assistant{end_token}\n\n", add_special_tokens=False)

    user_start = find_subsequence(input_ids.tolist(), user_tokens)
    assistant_start = find_subsequence(input_ids[user_start:].tolist(), assistant_tokens)
    
    if assistant_start != -1:
        assistant_start += user_start + len(assistant_tokens)
        labels[:assistant_start] = -100
        
    # mask loss on padding using attention mask
    labels[attention_mask == 0] = -100

    # Print the labelled and unlabeled tokens
    # labelled_tokens = tokenizer.decode(input_ids[labels != -100])
    # unlabelled_tokens = tokenizer.decode(input_ids[labels == -100])
    # print(f"Labelled tokens: {repr(labelled_tokens)} -- Unlabelled tokens: {repr(unlabelled_tokens)} \n ---------------------------------------- \n")

    labels = torch.cat([torch.full((num_image_tokens,), -100), labels])
    
    return input_ids, attention_mask, labels

def preprocess_image(image, image_processor):
    return image_processor(images=image, return_tensors="pt").pixel_values.squeeze()

class ImageTextData(IterableDataset):
    def __init__(self, data_dir, dataset_type, tokenizer, image_processor, max_length, num_image_tokens, caption_key=None, rank=0, world_size=1, base_seed=42, total_samples=1000000):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.num_image_tokens = num_image_tokens
        self.caption_key = caption_key
        self.rank = rank
        self.world_size = world_size
        self.base_seed = base_seed
        self.data_files = sorted(glob.glob(os.path.join(data_dir, "*.tar")))
        random.seed(self.base_seed)
        random.shuffle(self.data_files)
        self.total_samples = total_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = self.rank * (self.total_samples // self.world_size)
            iter_end = (self.rank + 1) * (self.total_samples // self.world_size)
        else:
            per_worker = int(math.ceil((self.total_samples // self.world_size) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.rank * (self.total_samples // self.world_size) + worker_id * per_worker
            iter_end = min(iter_start + per_worker, (self.rank + 1) * (self.total_samples // self.world_size))
        
        print(f"Rank {self.rank}, Worker {worker_info.id if worker_info else 'N/A'} iterating from {iter_start} to {iter_end}")

        sample_count = 0

        for tar_file in self.data_files:
            if sample_count >= iter_end:
                break

            with tarfile.open(tar_file, 'r') as tar:
                for member in tar.getmembers():
                    if member.name.endswith(('.jpg', '.png', '.jpeg')):
                        if sample_count >= iter_end:
                            break
                        if sample_count < iter_start:
                            sample_count += 1
                            continue
                        
                        image_file = tar.extractfile(member)
                        image = Image.open(io.BytesIO(image_file.read()))
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                                                    
                        json_member = tar.getmember(member.name.rsplit('.', 1)[0] + '.json')
                        json_file = tar.extractfile(json_member)
                        json_data = json.load(json_file)

                        preprocessed_image = preprocess_image(image, self.image_processor)
                                                
                        if self.dataset_type == "cauldron":
                            preprocessed_text = preprocess_cauldron_sample(json_data, self.tokenizer, self.max_length, self.num_image_tokens)
                        elif self.dataset_type == "caption":
                            preprocessed_text = preprocess_caption_sample(json_data, self.tokenizer, self.max_length, self.num_image_tokens, self.caption_key)
                        else:
                            raise ValueError("Invalid dataset_type. Choose 'cauldron' or 'caption'.")
                        
                        yield preprocessed_image, preprocessed_text
                        sample_count += 1

def create_dataloader(args, tokenizer, image_processor, num_image_tokens, epoch, world_size, rank, base_seed, total_samples):
    dataset = ImageTextData(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=args.max_length,
        num_image_tokens=num_image_tokens,
        caption_key=args.caption_key,
        rank=rank,
        world_size=world_size,
        base_seed=base_seed,
        total_samples=total_samples
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )

    return dataloader
