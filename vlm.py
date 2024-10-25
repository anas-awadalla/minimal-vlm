import torch
import torch.nn as nn
from transformers import AutoModel, SiglipVisionModel
from torch.nn import functional as F
import copy
from torch.nn import CrossEntropyLoss
import types
from typing import Optional, List, Union, Tuple
from dataclasses import dataclass
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from peft import get_peft_model, LoraConfig, TaskType

@dataclass
class VLMMoEOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    load_balancing_loss: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale

        # attention
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(self, dim, dim_out, num_latents=64, dim_head=64, heads=8, depth=1):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim_out))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(PerceiverAttention(dim=dim_out, dim_head=dim_head, heads=heads))
        self.norm = nn.LayerNorm(dim_out)
        self.to_out = nn.Linear(dim, dim_out)

    def forward(self, x):
        b = x.shape[0]
        latents = repeat(self.latents, 'n d -> b n d', b=b)
        x = self.to_out(x)  # Project to dim_out before attention
        
        for attn in self.layers:
            latents = attn(x, latents) + latents

        latents = self.norm(latents)
        return latents

class VLM(nn.Module):
    def __init__(self, lm, vision_encoder_name, use_perceiver=True, use_lora=False):
        super().__init__()
        # Language model
        self.lm = lm
        # Vision encoder
        self.vision_encoder = SiglipVisionModel.from_pretrained(vision_encoder_name)
        # Flag to use Perceiver Resampler
        self.use_perceiver = use_perceiver
        
        if use_perceiver:
            # Perceiver Resampler
            self.perceiver = PerceiverResampler(
                dim=self.vision_encoder.config.hidden_size,
                dim_out=self.lm.config.hidden_size,
                num_latents=64,  # You can adjust this
                depth=1  # Single attention layer
            )
        else:
            # Projection layer to align vision and language embeddings
            self.projection = nn.Linear(
                self.vision_encoder.config.hidden_size,
                self.lm.config.hidden_size
            )
        
        # Freeze vision encoder weights
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # Apply LoRA if specified
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=64,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
            )
            self.lm = get_peft_model(self.lm, lora_config)
                
        # Print parameter statistics on initialization
        self.print_param_stats()

    def process_images(self, pixel_values):
        image_embeddings = self.vision_encoder(pixel_values).last_hidden_state
        print("Image embeddings shape:", image_embeddings.shape)
        if self.use_perceiver:
            return self.perceiver(image_embeddings)
        else:
            return self.projection(image_embeddings)

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        batch_size = input_ids.size(0)
        if pixel_values is not None:
            image_embeddings = self.process_images(pixel_values)
            input_embeddings = self.lm.get_input_embeddings()(input_ids)
            combined_embeddings = torch.cat([image_embeddings, input_embeddings], dim=1)
            if attention_mask is not None:
                # Adjust attention mask to account for all image embeddings
                image_attention = torch.ones(batch_size, image_embeddings.size(1), device=attention_mask.device)
                attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        else:
            combined_embeddings = self.lm.get_input_embeddings()(input_ids)

        outputs = self.lm(inputs_embeds=combined_embeddings, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        
        if labels is not None:
            loss = outputs.loss

        return logits, loss

    def generate(self, input_ids, pixel_values, attention_mask=None, max_new_tokens=100, num_beams=4, temperature=1.0, **kwargs):
        batch_size = input_ids.size(0)
        if pixel_values is not None:
            image_embeddings = self.process_images(pixel_values)
            input_embeddings = self.lm.get_input_embeddings()(input_ids)
            combined_embeddings = torch.cat([image_embeddings, input_embeddings], dim=1)
            if attention_mask is not None:
                # Adjust attention mask to account for all image embeddings
                image_attention = torch.ones(batch_size, image_embeddings.size(1), device=attention_mask.device)
                attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        else:
            combined_embeddings = self.lm.get_input_embeddings()(input_ids)

        outputs = self.lm.generate(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            **kwargs
        )
        return outputs

    def print_param_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        new_params = trainable_params - sum(p.numel() for p in self.lm.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"VLM Model Statistics:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"New parameters: {new_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")

class ExpertFFN(nn.Module):
    def __init__(self, original_ffn, num_experts, hidden_size):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([self._clone_ffn(original_ffn) for _ in range(num_experts)])
        self.router = nn.Linear(hidden_size, num_experts)
        # init with std 0.02
        self.router.weight.data.normal_(mean=0.0, std=0.02)
    
    def _clone_ffn(self, original_ffn):
        return copy.deepcopy(original_ffn)

    def forward(self, x):
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        top_k_probs, top_k_indices = router_probs.topk(1, dim=-1)
        
        expert_outputs = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).float()
            expert_outputs += mask * expert(x)
        
        return expert_outputs, router_probs, router_logits
    
class MoEDecoderLayer(nn.Module):
    def __init__(self, original_layer, num_experts=4, alpha=0.01, beta=0.01):
        super().__init__()
        self.input_layernorm = original_layer.input_layernorm
        self.self_attn = original_layer.self_attn
        self.post_attention_layernorm = original_layer.post_attention_layernorm
        self.mlp = ExpertFFN(original_layer.mlp, num_experts, original_layer.mlp.hidden_size)
        self.num_experts = num_experts
        self.alpha = alpha
        self.beta = beta
        self.load_balancing_loss = 0
        self.router_z_loss = 0

    def compute_load_balancing_loss(self, router_probs):
        # Using load balancing loss from OLMO-MoE paper
        # L_LB = number of experts * sum(f_i * P_i) where f_i is the fraction of tokens routed to expert i 
        # and P_i is the total routing probability allocated to expert i
        # router_probs is of shape (batch_size, num_tokens, num_experts)
        loss = torch.zeros(1, device=router_probs.device)
        chosen_tokens = torch.argmax(router_probs, dim=-1)
        total_tokens = router_probs.size(0) * router_probs.size(1)  # batch_size * num_tokens

        for i in range(self.num_experts):
            frac_for_expert = (chosen_tokens == i).sum().float() / total_tokens
            total_prob_for_expert = router_probs[:, :, i].sum()
            loss += frac_for_expert * total_prob_for_expert
        loss *= self.num_experts
        return loss

    def compute_router_z_loss(self, router_logits):
        return torch.logsumexp(router_logits, dim=-1).pow(2).mean() * self.num_experts

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, 
                output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected with MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_probs, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        self.load_balancing_loss = self.compute_load_balancing_loss(router_probs)
        self.router_z_loss = self.compute_router_z_loss(router_logits)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class VLMMoE(VLM):
    def __init__(self, lm, vision_encoder_name, num_experts=4, alpha=0.01, beta=0.01, use_perceiver=False, use_lora=False):
        super().__init__(lm, vision_encoder_name, use_perceiver, use_lora)
        self.num_experts = num_experts
        self.alpha = alpha
        self.beta = beta
        self._upcycle_layers_to_moe()
        self._replace_lm_forward()
        setattr(self.lm, "alpha", alpha)
        setattr(self.lm, "beta", beta)
        
        # Print parameter statistics on initialization
        self.print_param_stats()

    def _upcycle_layers_to_moe(self):
        for i, layer in enumerate(self.lm.model.layers):
            self.lm.model.layers[i] = MoEDecoderLayer(layer, self.num_experts, 
                                                      self.alpha, self.beta)

    def _replace_lm_forward(self):
        def new_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: int = 0,
        ) -> Union[Tuple, VLMMoEOutput]:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

            loss = None
            if labels is not None:
                logits = logits.float()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            # Add MoE losses
            total_load_balancing_loss = sum(layer.load_balancing_loss for layer in self.model.layers if hasattr(layer, 'load_balancing_loss')) / len(self.model.layers)
            total_router_z_loss = sum(layer.router_z_loss for layer in self.model.layers if hasattr(layer, 'router_z_loss')) / len(self.model.layers)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output + (total_load_balancing_loss, total_router_z_loss) if loss is not None else output

            return VLMMoEOutput(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                load_balancing_loss=total_load_balancing_loss,
                router_z_loss=total_router_z_loss,
            )

        self.lm.forward = types.MethodType(new_forward, self.lm)

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        batch_size = input_ids.size(0)
        if pixel_values is not None:
            image_embeddings = self.process_images(pixel_values)
            input_embeddings = self.lm.get_input_embeddings()(input_ids)
            combined_embeddings = torch.cat([image_embeddings, input_embeddings], dim=1)
            if attention_mask is not None:
                image_attention = torch.ones(batch_size, image_embeddings.size(1), device=attention_mask.device)
                attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        else:
            combined_embeddings = self.lm.get_input_embeddings()(input_ids)

        outputs = self.lm(inputs_embeds=combined_embeddings, attention_mask=attention_mask, labels=labels)
        return outputs.logits, outputs.loss, outputs.load_balancing_loss, outputs.router_z_loss

    def print_param_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculate new parameters (MoE-specific parameters)
        moe_params = sum(p.numel() for layer in self.lm.model.layers 
                         if isinstance(layer.mlp, ExpertFFN) 
                         for p in layer.mlp.parameters())
        
        perceiver_params = sum(p.numel() for p in self.perceiver.parameters()) if self.use_perceiver else 0
        
        print(f"VLMMoE Model Statistics:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"New parameters (MoE): {moe_params:,}")
        print(f"Perceiver parameters: {perceiver_params:,}")

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model vlm \
#     --model_args pretrained="/mmfs1/gscratch/xlab/anasa2/datacomp-cauldron-model-1m-qwen-0.5b-so400m-4gpu-instruct-linear-proj/VLM_model_final.pt" \
#     --tasks textvqa ai2d chartqa mme mmvet textcaps ocrbench pope mmbench_en \
#     --batch_size 48 \
#     --log_samples \
#     --log_samples_suffix vlm_cogvlm_wsd_1m_384_bf16_8gpus_IT \
#     --output_path ./logs/

# Example usage:
# vlm = VLM(lm, vision_encoder, use_perceiver=True)
# vlm_moe = VLMMoE(lm, vision_encoder, num_experts=4, use_perceiver=True)

