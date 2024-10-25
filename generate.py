import argparse
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor
from vlm import VLM
from vision_encoder import SiglipPatchEncoder

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text based on an image and prompt")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained VLM model")
    parser.add_argument("--lm_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Language model name")
    parser.add_argument("--vision_encoder_name", type=str, default="google/siglip-so400m-patch14-384", help="Vision encoder name")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum length of generated text")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search")
    return parser.parse_args()

def load_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and image processor
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = AutoImageProcessor.from_pretrained(args.vision_encoder_name)

    # Load language model and vision encoder
    lm = AutoModelForCausalLM.from_pretrained(args.lm_name)

    # Initialize and load the VLM model
    model = VLM(lm, args.vision_encoder_name, use_perceiver=True)
    model.load_state_dict(torch.load(args.model_path, map_location=device)["model"])
    model.to(device)
    model.eval()

    while True:
        # Get image URL input
        image_url = input("Enter the image URL (or 'quit' to exit): ")
        if image_url.lower() == 'quit':
            break

        try:
            # Load and process the image
            image = load_image(image_url)
            pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)

            # Get text prompt input
            prompt = input("Enter the text prompt: ")

            # Determine tokens based on model name
            if "qwen" in args.lm_name.lower():
                start_token, end_token = "<|im_start|>", None
                eot_token = "<|im_end|>"
            else:
                start_token, end_token = "<|start_header_id|>", "<|end_header_id|>"
                eot_token = "<|eot_id|>"

            # Tokenize the prompt
            if "qwen" in args.lm_name.lower():
                formatted_prompt = f"{start_token}user\nDescribe this image in detail.{eot_token}{start_token}assistant\n{prompt}"
            else:
                formatted_prompt = f"{start_token}user{end_token if end_token is not None else ''}\n\nDescribe this image in detail.{eot_token}\n\n{start_token}assistant{end_token if end_token is not None else ''}\n\n{prompt}"
            input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids)

            # Generate text
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )

            # Decode the generated text
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print("\nGenerated text:")
            print(generated_text)
            print("\n" + "-"*50 + "\n")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again with a valid image URL.")
            print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()