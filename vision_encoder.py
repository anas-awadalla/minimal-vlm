import torch
from transformers import SiglipVisionModel

class SiglipPatchEncoder(torch.nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        
        # Load the vision model
        vision_model = SiglipVisionModel.from_pretrained(model_name_or_path)
        
        # Extract only the necessary components
        self.config = vision_model.config
        self.embeddings = vision_model.vision_model.embeddings
        self.encoder = vision_model.vision_model.encoder
        self.post_layernorm = vision_model.vision_model.post_layernorm

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        return last_hidden_state
        

# Usage example:
# model = SiglipPatchEncoder("google/siglip-base-patch16-224")
# pixel_values = torch.randn(1, 3, 224, 224)
# patch_encodings = model(pixel_values)