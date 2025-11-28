#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, GenerationMixin
from .stablelm.configuration_stablelm_epoch import StableLMEpochConfig
from .stablelm.modeling_stablelm_epoch import StableLMEpochModel, StableLMEpochForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from deepspeed.moe.layer import MoE
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import logger
from transformers.utils import ModelOutput
import os, json  # added for local saving of SiLU stats

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class MoELLaVAStablelmConfig(StableLMEpochConfig):
    model_type = "moe_llava_stablelm"

    def __init__(self,
                 moe_enable=True,
                 moe_mode='sparse',
                 moe_layers_idx=None,
                 ep_size=1,
                 top_k_experts=2,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 use_residual=False,
                 router_aux_loss_coef=0.01,
                 pretraining_tp=1,
                 **kwargs):
        
        # 设置默认值
        kwargs.setdefault('pretraining_tp', pretraining_tp)
        
        self.moe = dict(
            moe_enable=moe_enable,
            moe_mode=moe_mode,
            moe_layers_idx=moe_layers_idx,
            ep_size=ep_size,
            top_k_experts=top_k_experts,
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            min_capacity=min_capacity,
            use_residual=use_residual,
            router_aux_loss_coef=router_aux_loss_coef,
            train_modules=[
                # 'up_proj', 'down_proj', 'gate_proj', 'wg',
                # 'embed_tokens', 'lm_head'
            ]
        )

        super(MoELLaVAStablelmConfig, self).__init__(**kwargs)
        

class RePaMoELLaVAStablelmConfig(MoELLaVAStablelmConfig):
    model_type = "repa_moe_llava_stablelm"

    def __init__(self,
                 reparamed=False,
                 gated_ratio=1.0,
                 **kwargs):
        
        self.reparam = dict(
            reparamed=reparamed,
            target_gated_ratio=gated_ratio,
            current_gated_ratio=1.0,
        )

        super(RePaMoELLaVAStablelmConfig, self).__init__(**kwargs)


class MoELLaVAStablelmModel(LlavaMetaModel, StableLMEpochModel):
    config_class = MoELLaVAStablelmConfig

    def __init__(self, config: StableLMEpochConfig):
        super(MoELLaVAStablelmModel, self).__init__(config)


@dataclass
class MoEBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoECausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    moe_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None


def MoEStablelmDecoderLayer_forward(self):
    def forward(
            # self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            # padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # import ipdb
        # ipdb.set_trace()
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            # padding_mask=padding_mask,  # unuseful but conflict to flashattn
        )
        hidden_states = residual + hidden_states
        

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # import ipdb
        # ipdb.set_trace()
        moe_losses = []
        if len(hidden_states) == 3:
            moe_losses.append(hidden_states[1])
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (moe_losses,)

        return outputs

    return forward



def MoEStablelmModel_forward(self):
    def forward(
            # self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            output_moe_loss: Optional[bool] = True,
    ) -> Union[Tuple, MoEBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )
        
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # Process past_key_values length for the latest transformers pkg
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_key_values_length = past_key_values.get_seq_length()
            elif hasattr(past_key_values, "layers"):
                past_key_values_length = past_key_values.layers[0].get_seq_length()
            elif isinstance(past_key_values, tuple):
                past_key_values_length = past_key_values[0][0].shape[2]
        
            seq_length_with_past = seq_length + past_key_values_length
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            # 确保 position_ids 的形状与当前序列长度匹配
            if position_ids.numel() == seq_length * batch_size:
                position_ids = position_ids.view(-1, seq_length).long()
            else:
                """
                首次推理:
                input_ids: [batch_size, full_sequence_length]
                position_ids: [0]
                """
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                if position_ids.numel() == batch_size:
                    # 如果 position_ids 是一维的，说明是第一次输入
                    position_ids = torch.arange(
                        past_key_values_length,
                        seq_length + past_key_values_length,
                        dtype=torch.long,
                        device=device,
                    )
                    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
                else:
                    assert False, f"position_ids shape mismatches"
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Embed positions
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past),
                    dtype=torch.bool,
                    device=inputs_embeds.device,
                )
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_moe_loss = [] if output_moe_loss else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if past_key_values is not None:
                if hasattr(past_key_values, "layers"):
                    past_key_value = past_key_values.layers[idx]
                else:
                    past_key_value = past_key_values[idx]
            else:
                past_key_value = None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_moe_loss:
                all_moe_loss.extend(layer_outputs[-1])

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        if use_cache:
            if hasattr(past_key_values, "layers"):
                for i, cache in enumerate(next_decoder_cache):
                    past_key_values.layers[i] = cache
                next_cache = past_key_values
            else:
                next_cache = next_decoder_cache
        else:
            next_cache = None
            
        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_moe_loss] if
                v is not None)
        return MoEBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            moe_loss_list=all_moe_loss,
        )

    return forward


class MoELLaVAStablelmForCausalLM(StableLMEpochForCausalLM, LlavaMetaForCausalLM, GenerationMixin):
    config_class = MoELLaVAStablelmConfig

    def __init__(self, config):
        super(StableLMEpochForCausalLM, self).__init__(config)
        self.model = MoELLaVAStablelmModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
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
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoECausalLMOutputWithPast]:
        # print('before prepare_inputs_labels_for_multimodal')
        # import ipdb
        # ipdb.set_trace()
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
        # import ipdb
        # ipdb.set_trace()
        # print('after prepare_inputs_labels_for_multimodal')
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
        )
        # import ipdb
        # ipdb.set_trace()
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        moe_loss, moe_losses = None, []
        if len(outputs[-1]) > 0:
            moe_loss_list = outputs[-1]
            # import ipdb
            # ipdb.set_trace()
            for moe_loss in moe_loss_list:
                if moe_loss is not None:
                    moe_losses.append(moe_loss)
            moe_loss = self.router_aux_loss_coef * sum(moe_losses)
            if labels is not None:
                # print("Losses:", loss, sum(moe_losses), loss + moe_loss)
                loss += moe_loss
        # import ipdb
        # ipdb.set_trace()
        if not return_dict:
            output = (logits,) + outputs[1:]
            output = (moe_loss,) + output if moe_loss is not None else output
            return (loss,) + output if loss is not None else output

        return MoECausalLMOutputWithPast(
            loss=loss,
            moe_loss=moe_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            moe_loss_list=outputs.moe_loss_list,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

    def initialize_moe_modules(self, model_args):
        self.config.moe['moe_enable'] = model_args.moe_enable
        self.config.moe['train_modules'] = model_args.train_modules
        self.config.moe['moe_mode'] = model_args.moe_mode
        self.config.moe['moe_layers_idx'] = model_args.moe_layers_idx
        self.config.moe['ep_size']= model_args.ep_size
        self.config.moe['top_k_experts'] = model_args.top_k_experts
        self.config.moe['capacity_factor'] = model_args.capacity_factor
        self.config.moe['eval_capacity_factor'] = model_args.eval_capacity_factor
        self.config.moe['min_capacity'] = model_args.min_capacity
        self.config.moe['use_residual'] = model_args.use_residual
        self.config.moe['router_aux_loss_coef'] = self.router_aux_loss_coef = model_args.router_aux_loss_coef
        # self.config.moe['train_modules'] = [
        #         # 'mlp.w1', 'mlp.w2', 'mlp.c_proj', 'wg',
        #         # 'wte', 'lm_head'
        #     ]
        if self.config.moe['train_modules'] is not None and len(self.config.moe['train_modules']) > 0:
            for n, p in self.named_parameters():
                if any(name in n for name in self.config.moe['train_modules']):
                    continue
                else:
                    p.requires_grad = False
        
        num_layers = self.config.num_hidden_layers

        moe_layers_idx = model_args.moe_layers_idx
        if model_args.moe_layers_idx is not None:
            model_args.moe_mode = 'custom'
            assert len(model_args.moe_layers_idx) <= num_layers
            assert max(model_args.moe_layers_idx) < num_layers
            assert min(model_args.moe_layers_idx) >= 0
        else:
            if model_args.moe_mode == "first_half":
                moe_layers_idx = list(range(0, num_layers // 2))
            elif model_args.moe_mode == "second_half":
                moe_layers_idx = list(range(num_layers // 2, num_layers))
            elif model_args.moe_mode == "sparse":
                moe_layers_idx = list(range(num_layers))[::2]
            elif model_args.moe_mode == "dense":
                moe_layers_idx = list(range(num_layers))
            else:
                raise NotImplementedError(
                    f'Only support ["first_half", "second_half", "sparse", "dense"], but found {model_args.moe_mode}')

        self.config.moe['moe_layers_idx'] = moe_layers_idx
        if len(model_args.num_experts) == 1:
            self.config.moe['num_experts'] = model_args.num_experts * len(moe_layers_idx)
        assert len(self.config.moe['num_experts']) == len(moe_layers_idx)

        # Helper: attach a forward wrapper to compute pairwise cosine similarity between expert outputs
        def _attach_expert_similarity_hook(moe_module):
            if not hasattr(moe_module, 'deepspeed_moe') or not hasattr(moe_module.deepspeed_moe, 'experts'):
                return
            if getattr(moe_module, '_similarity_hook_attached', False):
                return
            moe_module._similarity_hook_attached = True
            _orig_forward = moe_module.forward

            def _forward_with_similarity(x, *args, **kwargs):
                out = _orig_forward(x, *args, **kwargs)
                try:
                    # Token-wise expert output similarity BEFORE gating weighting
                    sample = x
                    if isinstance(sample, (tuple, list)) and len(sample) > 0:
                        sample = sample[0]
                    if sample.dim() >= 2:
                        tokens = sample.reshape(-1, sample.size(-1))
                    else:
                        tokens = sample
                    max_tokens = 64
                    if tokens.size(0) > max_tokens:
                        tokens = tokens[:max_tokens]
                    experts = moe_module.deepspeed_moe.experts.deepspeed_experts
                    if len(experts) >= 2 and tokens.numel() > 0:
                        expert_outputs = []  # list of [T, H]
                        for expert in experts:
                            y = expert(tokens)
                            expert_outputs.append(y)
                        stacked = torch.stack(expert_outputs, dim=0)  # [E, T, H]
                        stacked = F.normalize(stacked, dim=2)
                        per_token = stacked.permute(1, 0, 2)  # [T, E, H]
                        sims = torch.matmul(per_token, per_token.transpose(1, 2))  # [T, E, E]
                        sim_mean = sims.mean(dim=0)  # [E, E]
                        moe_module.expert_cosine_similarity = sim_mean.detach().to('cpu')
                except Exception:
                    pass
                return out

            moe_module.forward = _forward_with_similarity

        for num_experts, layer_num in zip(self.config.moe['num_experts'], moe_layers_idx):
            pretrained_state_dict = self.model.layers[layer_num].mlp.state_dict()
            self.model.layers[layer_num].mlp = MoE(
                self.config.hidden_size,
                expert=self.model.layers[layer_num].mlp,
                num_experts=num_experts,
                ep_size=model_args.ep_size,
                k=model_args.top_k_experts,
                capacity_factor=model_args.capacity_factor,
                eval_capacity_factor=model_args.eval_capacity_factor,
                min_capacity=model_args.min_capacity,
                use_residual=model_args.use_residual,
            )
            
            # Attach similarity hook
            # _attach_expert_similarity_hook(self.model.layers[layer_num].mlp)
            
            for e in self.model.layers[layer_num].mlp.deepspeed_moe.experts.deepspeed_experts:  # check weight
                loaded_state_dict = e.state_dict()
                assert all([torch.allclose(pretrained_state_dict[k], v) for k, v in loaded_state_dict.items()])
                assert all([torch.allclose(loaded_state_dict[k], v) for k, v in pretrained_state_dict.items()])
                
        # ipdb.set_trace()
        rank0_print(f"LLM num_layers: {num_layers}, MoE num_layers: {len(moe_layers_idx)}, where\n",
                    *[f'layer-{layer_num} has {num_experts} experts\n' for num_experts, layer_num in
                      zip(self.config.moe['num_experts'], moe_layers_idx)])

        for m in self.model.layers:
            m.forward = MoEStablelmDecoderLayer_forward(m)
        rank0_print(f'replace StablelmDecoderLayer.forward to MoEStablelmDecoderLayer.forward')
        self.model.forward = MoEStablelmModel_forward(self.model)
        rank0_print(f'replace StablelmModel.forward to MoEStablelmModel.forward')
        # ipdb.set_trace()



class EvalMoELLaVAStablelmForCausalLM(MoELLaVAStablelmForCausalLM, GenerationMixin):
    config_class = MoELLaVAStablelmConfig

    def __init__(self, config):
        super(EvalMoELLaVAStablelmForCausalLM, self).__init__(config)

        self.router_aux_loss_coef = self.config.moe['router_aux_loss_coef']
        num_layers = self.config.num_hidden_layers
        moe_layers_idx = self.config.moe['moe_layers_idx']

        # For SiLU channel stats aggregation and saving
        self._silu_channel_values = {}
        self._silu_stats_path = os.environ.get('SILU_STATS_PATH', 'silu_stats_eval.json')
        # For expert similarity aggregation and saving
        self._expert_similarity = {}
        self._expert_sim_path = os.environ.get('EXPERT_SIM_PATH', 'expert_similarity_eval.json')

        # Helper reused in eval: attach similarity hook
        def _attach_expert_similarity_hook(moe_module, layer_idx: int):
            if not hasattr(moe_module, 'deepspeed_moe') or not hasattr(moe_module.deepspeed_moe, 'experts'):
                return
            if getattr(moe_module, '_similarity_hook_attached', False):
                return
            moe_module._similarity_hook_attached = True
            _orig_forward = moe_module.forward

            def _forward_with_similarity(x, *args, **kwargs):
                out = _orig_forward(x, *args, **kwargs)
                try:
                    sample = x
                    if isinstance(sample, (tuple, list)) and len(sample) > 0:
                        sample = sample[0]
                    if sample.dim() >= 2:
                        tokens = sample.reshape(-1, sample.size(-1))
                    else:
                        tokens = sample
                    max_tokens = 64
                    if tokens.size(0) > max_tokens:
                        tokens = tokens[:max_tokens]
                    experts = moe_module.deepspeed_moe.experts.deepspeed_experts
                    if len(experts) >= 2 and tokens.numel() > 0:
                        expert_outputs = []
                        for expert in experts:
                            y = expert(tokens)
                            expert_outputs.append(y)
                        stacked = torch.stack(expert_outputs, dim=0)  # [E, T, H]
                        stacked = F.normalize(stacked, dim=2)
                        per_token = stacked.permute(1, 0, 2)  # [T, E, H]
                        sims = torch.matmul(per_token, per_token.transpose(1, 2))  # [T, E, E]
                        sim_mean = sims.mean(dim=0)  # [E, E]
                        moe_module.expert_cosine_similarity = sim_mean.detach().to('cpu')
                        # save into class dict as {layer_a: {"similarity": [[..]]}}
                        layer_key = f"layer_{layer_idx}"
                        self._expert_similarity[layer_key] = {"similarity": moe_module.expert_cosine_similarity.tolist()}
                except Exception:
                    pass
                return out

            moe_module.forward = _forward_with_similarity

        # New: attach SiLU activation capture to experts of each MoE layer
        def _attach_silu_capture_to_experts(moe_module, layer_idx: int):
            if not hasattr(moe_module, 'deepspeed_moe') or not hasattr(moe_module.deepspeed_moe, 'experts'):
                return
            experts = moe_module.deepspeed_moe.experts.deepspeed_experts
            for expert_idx, expert in enumerate(experts):
                if getattr(expert, '_silu_capture_attached', False):
                    continue
                expert._silu_capture_attached = True
                # keep original forward
                _orig_expert_forward = expert.forward

                def _expert_forward_with_silu_capture(x, *args, __layer_idx=layer_idx, __expert_idx=expert_idx, **kwargs):
                    # Capture post-SiLU gate activation aggregated across all channels (fp16)
                    try:
                        if hasattr(expert, 'gate_proj') and hasattr(expert, 'act_fn'):
                            gate = expert.act_fn(expert.gate_proj(x))
                            # detach, move to cpu, and flatten tokens/channels
                            gate2d = gate.detach().to('cpu')
                            if gate2d.dim() > 2:
                                gate2d = gate2d.view(-1, gate2d.size(-1))  # [T, C]
                            # Quantize to float16 before storing and flatten across all channels
                            gate2d = gate2d.to(dtype=torch.float16)
                            flat = gate2d.reshape(-1)
                            # store aggregated values per expert
                            layer_key = f"layer_{__layer_idx}"
                            expert_key = f"expert_{__expert_idx}"
                            leaf = self._silu_channel_values.setdefault(layer_key, {}).setdefault(expert_key, [])
                            # extend with fp16-quantized values as Python floats
                            leaf.extend([float(v) for v in flat.tolist()])
                    except Exception:
                        pass
                    return _orig_expert_forward(x, *args, **kwargs)

                expert.forward = _expert_forward_with_silu_capture

        for num_experts, layer_num in zip(self.config.moe['num_experts'], moe_layers_idx):
            self.model.layers[layer_num].mlp = MoE(
                self.config.hidden_size,
                expert=self.model.layers[layer_num].mlp,
                num_experts=num_experts,
                ep_size=self.config.moe['ep_size'],
                k=self.config.moe['top_k_experts'],
                capacity_factor=self.config.moe['capacity_factor'],
                eval_capacity_factor=self.config.moe['eval_capacity_factor'],
                min_capacity=self.config.moe['min_capacity'],
                use_residual=self.config.moe['use_residual'],
            )
            # Attach similarity hook
            _attach_expert_similarity_hook(self.model.layers[layer_num].mlp, layer_num)
            # Attach SiLU capture hooks to experts
            _attach_silu_capture_to_experts(self.model.layers[layer_num].mlp, layer_num)
        print(f"LLM num_layers: {num_layers}, MoE num_layers: {len(moe_layers_idx)}, where\n",
                    *[f'layer-{layer_num} has {num_experts} experts\n' for num_experts, layer_num in
                      zip(self.config.moe['num_experts'], moe_layers_idx)])

        for m in self.model.layers:
            m.forward = MoEStablelmDecoderLayer_forward(m)
        print(f'replace StablelmDecoderLayer.forward to MoEStablelmDecoderLayer.forward')
        self.model.forward = MoEStablelmModel_forward(self.model)
        print(f'replace StablelmModel.forward to MoEStablelmModel.forward')

    def _save_silu_stats(self, path: Optional[str] = None):
        try:
            save_path = path or self._silu_stats_path
            # ensure directory exists if path includes folder
            dirn = os.path.dirname(save_path)
            if dirn and not os.path.exists(dirn):
                os.makedirs(dirn, exist_ok=True)
            # write json
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self._silu_channel_values, f)
            print(f"Saved SiLU stats to {save_path}")
        except Exception as e:
            print(f"Failed to save SiLU stats: {e}")

    def _save_expert_similarity(self, path: Optional[str] = None):
        try:
            save_path = path or self._expert_sim_path
            dirn = os.path.dirname(save_path)
            if dirn and not os.path.exists(dirn):
                os.makedirs(dirn, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self._expert_similarity, f)
            print(f"Saved expert similarity to {save_path}")
        except Exception as e:
            print(f"Failed to save expert similarity: {e}")

    # Override forward to save stats after inference step
    def forward(
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
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            return_dict=return_dict,
        )
        if not self.training:
            self._save_silu_stats()
            self._save_expert_similarity()
        return out



class RePaMLP(nn.Module):
    def __init__(self, config: StableLMEpochConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Reparameterization flag
        self.reparamed = config.reparam["reparamed"]
        
        if self.reparamed:
            # Ratio of active (non-masked) channels
            self.gated_ratio = config.reparam["current_gated_ratio"]
            
            if self.gated_ratio == 1.0:
                self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
                self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
                self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
                self.act_fn = nn.SiLU()
                self.repa_proj = None
            elif self.gated_ratio == 0.0:
                self.gate_proj = None
                self.up_proj = None
                self.down_proj = None
                self.act_fn = None
                self.repa_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            else:
                self.num_gated_channels = int(self.intermediate_size * self.gated_ratio)
                self.gate_proj = nn.Linear(config.hidden_size, self.num_gated_channels, bias=False)
                self.up_proj = nn.Linear(config.hidden_size, self.num_gated_channels, bias=False)
                self.down_proj = nn.Linear(self.num_gated_channels, config.hidden_size, bias=False)
                self.act_fn = nn.SiLU()
                self.repa_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        else:
            self.num_gated_channels = self.intermediate_size
            self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
            self.act_fn = nn.SiLU()
            self.repa_proj = None
            
            # mask semantics (IMPORTANT): mask == True means this channel IS MASKED (linear), mask == False means gated
            self.register_buffer('mask', torch.zeros(self.intermediate_size, dtype=torch.bool))  # start with no masked channels
            # Running mean statistics for active (unmasked) channels
            self.register_buffer('channel_sum', torch.zeros(self.intermediate_size))
            
            self.gate_scaler = nn.Parameter(torch.zeros(self.intermediate_size))
            self.up_scaler = nn.Parameter(torch.zeros(self.intermediate_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If reparameterized, use the reparameterized form. No need to track running means or masks.
        if self.reparamed:
            if self.repa_proj is not None and self.up_proj is not None and self.down_proj is not None and self.gate_proj is not None:
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) + self.repa_proj(x)
            elif self.repa_proj is not None:
                return self.repa_proj(x)
            else:
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        else:
            x_up = self.up_proj(x) # B, N, pC
            
            # Compute linear gate then SiLU for active path
            x_gate_linear = self.gate_proj(x) # B, N, pC
            x_gate_act = self.act_fn(x_gate_linear) # B, N, pC
            
            # Statistics: per-channel sum over active tokens only
            if self.training:
                with torch.no_grad():
                    flat = x_gate_act.reshape(-1, x_gate_act.size(-1))  # B*N, pC
                    valid_tokens = (flat.abs().sum(dim=1) > 1e-9)
                    if valid_tokens.any():
                        cur_sum = flat[valid_tokens].sum(dim=0)
                        # update running mean on active channels only
                        self.channel_sum = self.channel_sum + cur_sum

            x_gate_act_times_up = x_gate_act * x_up
            x_gate_linear_adds_up = x_gate_linear * self.gate_scaler + x_up * self.up_scaler
            
            x_gate_up = torch.where(
                self.mask[None, None, :], 
                x_gate_linear_adds_up,
                x_gate_act_times_up
            )
            
            x = self.down_proj(x_gate_up)

            return x

    def reparam(self):
        if not self.reparamed:
            linear_idx = self.mask.nonzero(as_tuple=False).flatten()
            nonlinear_idx = (~self.mask).nonzero(as_tuple=False).flatten()
            
            if linear_idx.numel() > 0:
                # Some channels are masked and weights can be linearly reparameterized
                gate_weight = (self.gate_proj.weight * self.gate_scaler[:, None])[self.mask, :]
                up_weight = (self.up_proj.weight * self.up_scaler[:, None])[self.mask, :]
                gate_up_weight = gate_weight + up_weight
                down_weight = self.down_proj.weight[:, self.mask]
                repa_weight = down_weight @ gate_up_weight
                self.repa_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.repa_proj.weight = nn.Parameter(repa_weight)
            else:
                self.repa_proj = None
                
            if nonlinear_idx.numel() > 0:
                # Some channels remain active and need gating and nonlinearity
                gate_proj_weight = self.gate_proj.weight[~self.mask, :]
                self.gate_proj = nn.Linear(self.hidden_size, nonlinear_idx.numel(), bias=False)
                self.gate_proj.weight = nn.Parameter(gate_proj_weight)

                up_proj_weight = self.up_proj.weight[~self.mask, :]
                self.up_proj = nn.Linear(self.hidden_size, nonlinear_idx.numel(), bias=False)
                self.up_proj.weight = nn.Parameter(up_proj_weight)

                down_proj_weight = self.down_proj.weight[:, ~self.mask]
                self.down_proj = nn.Linear(nonlinear_idx.numel(), self.hidden_size, bias=False)
                self.down_proj.weight = nn.Parameter(down_proj_weight)
            else:
                self.gate_proj = None
                self.up_proj = None
                self.down_proj = None
                
            self.reparamed = True
            self.num_gated_channels = nonlinear_idx.numel()
            self.mask = None
            self.channel_sum = None
            self.gate_scaler = None
            self.up_scaler = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    def adjust_gated_ratio(self, gated_ratio: float):
        """
        Adjust gating ratio by masking additional channels with smallest channel_sum values.
        Steps:
        1. Determine desired number of active channels = floor(ratio * total).
        2. Compute how many channels must be masked = total - active.
        3. If we need to mask MORE channels, select from currently unmasked channels those with lowest channel_sum.
        4. If we need to UNMASK channels (ratio increased), unmask channels with highest channel_sum first.
        """
        gated_ratio = float(gated_ratio)
        gated_ratio = min(1.0, max(0.0, gated_ratio))
        self.gated_ratio = gated_ratio
        desired_gated = int(self.intermediate_size * gated_ratio)
        desired_linear = self.intermediate_size - desired_gated
        current_linear = int(self.mask.sum().item())
        if desired_linear == 0:
            # Unmask all
            self.mask[:] = False
            self.num_gated_channels = self.intermediate_size

        elif desired_linear > current_linear:
            # Need to mask additional channels
            need_mask = desired_linear - current_linear
            candidates = (~self.mask).nonzero(as_tuple=False).flatten()
            if candidates.numel() > 0:
                # Sort candidates by channel_sum ascending (smallest first - least important channels)
                sums = self.channel_sum[candidates]
                _, order = torch.sort(sums)  # ascending
                select = candidates[order[:need_mask]] if need_mask < candidates.numel() else candidates
                self.mask[select] = True
                
        # Update counts
        self.num_gated_channels = desired_gated

    def init_scaler(self):
        self.gate_scaler = torch.zeros_like(self.gate_proj.weight)
        self.up_scaler = torch.zeros_like(self.up_proj.weight)


class RePaMoE(MoE):
    """
    RePaMoE (Reparametrized Mixture of Experts): 
    Inherits from DeepSpeed's MoE and adds reparameterization functionality.
    """
    def __init__(self, hidden_size, expert, num_experts=4, ep_size=1, k=2, 
                 capacity_factor=1.0, eval_capacity_factor=1.0, min_capacity=4, 
                 use_residual=False, gated_ratio=1.0, reparamed=False):
        # Initialize the parent MoE class with RePaMLP experts
        super().__init__(
            hidden_size=hidden_size,
            expert=expert,  # This will be a RePaMLP instance
            num_experts=num_experts,
            ep_size=ep_size,
            k=k,
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            min_capacity=min_capacity,
            use_residual=use_residual
        )
        
        # Adjust gated ratio for all experts if needed
        self.adjust_gated_ratio(gated_ratio)
        self.gated_ratio = gated_ratio

        # Initialize scaler for all experts
        self.init_scaler()

        self.reparamed = reparamed

        for expert in self.deepspeed_moe.experts.deepspeed_experts:
            if not isinstance(expert, RePaMLP):
                raise ValueError("Experts must be instances of RePaMLP for RePaMoE")
            for param in expert.parameters():
                param.allreduce = False  # Disable allreduce for expert params
        for param in self.deepspeed_moe.gate.parameters():
            param.allreduce = False # Disable allreduce for gate params
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through RePaMoE.
        If reparamed=False, use MoE's forward.
        If reparamed=True, combine MoE's forward with new expert's output.
        """
        # if self.reparamed and self.reparam_ffn is not None:
        #     # Get MoE output
        #     moe_output, moe_l_aux, moe_exp_counts = super().forward(x, *args, **kwargs)
        #     # Get new expert output
        #     reparamed_expert_output = self.reparam_ffn(x)
        #     # Combine outputs
        #     combined_output = moe_output + reparamed_expert_output
        #     return combined_output, moe_l_aux, moe_exp_counts 
        # else:
        #     # Use standard MoE forward
        #     return super().forward(x, *args, **kwargs)
        
        return super().forward(x, *args, **kwargs)
    
    def reparam(self):
        """
        Aggregate expert reparam results into a new expert.
        This method reparameterizes all experts and creates a new aggregated expert.
        """
        if not self.reparamed:
            # Access experts from the parent MoE class
            if hasattr(self, 'deepspeed_moe') and hasattr(self.deepspeed_moe, 'experts'):
                experts = self.deepspeed_moe.experts.deepspeed_experts
                for expert in experts:
                    if hasattr(expert, 'reparam') and callable(expert.reparam):
                        expert.reparam()
            self.reparamed = True
        
    def adjust_gated_ratio(self, gated_ratio: float):
        """Apply adjust_gated_ratio to all experts"""
        if hasattr(self, 'deepspeed_moe') and hasattr(self.deepspeed_moe, 'experts'):
            experts = self.deepspeed_moe.experts.deepspeed_experts
            for expert in experts:
                if hasattr(expert, 'adjust_gated_ratio') and callable(expert.adjust_gated_ratio):
                    expert.adjust_gated_ratio(gated_ratio)
        self.gated_ratio = gated_ratio

    def init_scaler(self):
        """Initialize the scaler for all experts"""
        if hasattr(self, 'deepspeed_moe') and hasattr(self.deepspeed_moe, 'experts'):
            experts = self.deepspeed_moe.experts.deepspeed_experts
            for expert in experts:
                if hasattr(expert, 'init_scaler') and callable(expert.init_scaler):
                    expert.init_scaler()



class RePaMoELLaVAStablelmForCausalLM(MoELLaVAStablelmForCausalLM, GenerationMixin):
    """
    RePaMoE version of LLaVA StableLM for Causal LM.
    Replaces MoE with RePaMoE and expert with RePaMLP, inheriting parameters.
    """
    config_class = RePaMoELLaVAStablelmConfig

    def __init__(self, config):
        super(RePaMoELLaVAStablelmForCausalLM, self).__init__(config)

        self.router_aux_loss_coef = self.config.moe['router_aux_loss_coef']
        num_layers = self.config.num_hidden_layers
        moe_layers_idx = self.config.moe['moe_layers_idx']
        
        # Replace MoE layers with RePaMoE after model initialization
        for num_experts, layer_num in zip(self.config.moe['num_experts'], moe_layers_idx):
            # Create RePaMLP from the original MLP
            repa_expert = RePaMLP(config)
            
            # Replace with RePaMoE
            self.model.layers[layer_num].mlp = RePaMoE(
                hidden_size=self.config.hidden_size,
                expert=repa_expert,
                num_experts=num_experts,
                ep_size=self.config.moe['ep_size'],
                k=self.config.moe['top_k_experts'],
                capacity_factor=self.config.moe['capacity_factor'],
                eval_capacity_factor=self.config.moe['eval_capacity_factor'],
                min_capacity=self.config.moe['min_capacity'],
                use_residual=self.config.moe['use_residual'],
                gated_ratio=self.config.reparam['current_gated_ratio'],
                reparamed=self.config.reparam['reparamed'],
            )
        
            # Attach similarity hook
            # _attach_expert_similarity_hook(self.model.layers[layer_num].mlp)
            
        rank0_print(f"LLM num_layers: {num_layers}, RePaMoE num_layers: {len(moe_layers_idx)}, where\n",
                    *[f'layer-{layer_num} has {num_experts} experts\n' for num_experts, layer_num in
                      zip(self.config.moe['num_experts'], moe_layers_idx)])

        for m in self.model.layers:
            m.forward = MoEStablelmDecoderLayer_forward(m)
        rank0_print(f'replace StablelmDecoderLayer.forward to MoEStablelmDecoderLayer.forward')
        self.model.forward = MoEStablelmModel_forward(self.model)
        rank0_print(f'replace StablelmModel.forward to MoEStablelmModel.forward')

    def get_model(self):
        return self.model

    def reparam_moe_layers(self):
        """
        Reparameterize all RePaMoE layers in the model.
        """
        moe_layers_idx = self.config.moe['moe_layers_idx']
        for layer_num in moe_layers_idx:
            moe_layer = self.model.layers[layer_num].mlp
            if isinstance(moe_layer, RePaMoE):
                moe_layer.reparam()
                print(f"Reparameterized RePaMoE layer {layer_num}")
            else:
                print(f"Layer {layer_num} is not a RePaMoE layer, skipping reparameterization")
        self.config.reparam["reparamed"] = True
            
    def adjust_gated_ratio_all_layers(self, gated_ratio: float):
        """
        Adjust gated ratio for all RePaMoE layers.
        """
        moe_layers_idx = self.config.moe['moe_layers_idx']
        for layer_num in moe_layers_idx:
            moe_layer = self.model.layers[layer_num].mlp
            if isinstance(moe_layer, RePaMoE):
                moe_layer.adjust_gated_ratio(gated_ratio)
        print(f"Adjusted gated ratio to {gated_ratio} for all RePaMoE layers")
        self.config.reparam["current_gated_ratio"] = gated_ratio
        
    def init_scaler(self):
        """
        Initialize the scaler for all RePaMoE layers.
        """
        moe_layers_idx = self.config.moe['moe_layers_idx']
        for layer_num in moe_layers_idx:
            moe_layer = self.model.layers[layer_num].mlp
            if isinstance(moe_layer, RePaMoE):
                moe_layer.init_scaler()
        print(f"Initialized scaler for all RePaMoE layers")

    def disable_moe_allreduce(self):
        """
        Disable allreduce for all parameters in MoE layers.
        This function sets allreduce=False for expert parameters and gate parameters 
        in all RePaMoE layers to prevent gradient synchronization across processes.
        """
        moe_layers_idx = self.config.moe['moe_layers_idx']
        for layer_num in moe_layers_idx:
            moe_layer = self.model.layers[layer_num].mlp
            if isinstance(moe_layer, RePaMoE):
                # Disable allreduce for expert parameters
                if hasattr(moe_layer, 'deepspeed_moe') and hasattr(moe_layer.deepspeed_moe, 'experts'):
                    experts = moe_layer.deepspeed_moe.experts.deepspeed_experts
                    for expert in experts:
                        for param in expert.parameters():
                            param.allreduce = False
                            param.group_name = moe_layer.expert_group_name

                # Disable allreduce for gate parameters
                if hasattr(moe_layer, 'deepspeed_moe') and hasattr(moe_layer.deepspeed_moe, 'gate'):
                    for param in moe_layer.deepspeed_moe.gate.parameters():
                        param.allreduce = False
                        param.group_name = moe_layer.expert_group_name

                # Disable allreduce for reparam_ffn parameters if exists
                if hasattr(moe_layer, 'reparam_ffn') and moe_layer.reparam_ffn is not None:
                    for param in moe_layer.reparam_ffn.parameters():
                        param.allreduce = False
                        param.group_name = moe_layer.expert_group_name
        
        print(f"Disabled allreduce for all MoE layer parameters in {len(moe_layers_idx)} layers")



AutoConfig.register("moe_llava_stablelm", MoELLaVAStablelmConfig)
AutoConfig.register("repa_moe_llava_stablelm", RePaMoELLaVAStablelmConfig)
AutoModelForCausalLM.register(MoELLaVAStablelmConfig, MoELLaVAStablelmForCausalLM)
AutoModelForCausalLM.register(MoELLaVAStablelmConfig, EvalMoELLaVAStablelmForCausalLM)
AutoModelForCausalLM.register(RePaMoELLaVAStablelmConfig, RePaMoELLaVAStablelmForCausalLM)