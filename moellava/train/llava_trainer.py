import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    # ALL_LAYERNORM_LAYERS,
    # ShardedDDPOption,
    logger,
)
from moellava.constants import IGNORE_INDEX

# Replace the string list with a tuple of actual layer norm classes
def _customized_layer_norm_types():
    types = [nn.LayerNorm]
    try:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        types.append(LlamaRMSNorm)
    except Exception:
        pass
    try:
        # Some versions expose a generic RMSNorm
        from transformers.models.llama.modeling_llama import RMSNorm as LlamaRMSNormBase
        types.append(LlamaRMSNormBase)
    except Exception:
        pass
    try:
        from transformers.models.t5.modeling_t5 import T5LayerNorm
        types.append(T5LayerNorm)
    except Exception:
        pass
    try:
        # NeoX-style norm (if present)
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayerNorm
        types.append(GPTNeoXLayerNorm)
    except Exception:
        pass
    # de-dup
    uniq = []
    for t in types:
        if t not in uniq:
            uniq.append(t)
    return tuple(uniq)

ALL_LAYERNORM_LAYERS = _customized_layer_norm_types()

from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self, dataset) -> Optional[torch.utils.data.Sampler]:
        if dataset is None or not has_length(dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler(dataset)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize RePaMoE fine-tuning state
        if hasattr(self.args, 'finetune_repa_mode') and self.args.finetune_repa_mode:
            self.repa_state = {
                'stage_1_steps': 0,
                'stage_2_steps': 0,
                'current_stage': 1,
                'stage_1_complete': False,
                'reparam_called': False,
                'initial_gated_ratio': 1.0,
                'target_gated_ratio': getattr(self.args, 'gated_ratio', 0.25),
                'total_training_steps': 0,
                'current_gated_ratio': 1.0,
                'moe_layers_idx': [],
                'has_repamoe': False
            }
            # Setup RePaMoE fine-tuning mode if enabled
            self.setup_repa_finetuning()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Standard loss from model (includes CE and moe aux if any)
        outputs = model(**inputs)
        loss = outputs['loss'] if isinstance(outputs, dict) else getattr(outputs, 'loss', None)
        if loss is None:
            loss = outputs[0]
        
        # KD integration
        try:
            if getattr(self.args, 'finetune_repa_mode', False) and getattr(self.args, 'self_kd', False):
                teacher = getattr(model, 'teacher_model', None)
                if teacher is not None and 'labels' in inputs and inputs['labels'] is not None:
                    with torch.no_grad():
                        t_outputs = teacher(
                            input_ids=inputs.get('input_ids'),
                            attention_mask=inputs.get('attention_mask'),
                            position_ids=inputs.get('position_ids'),
                            inputs_embeds=inputs.get('inputs_embeds'),
                            images=inputs.get('images', None),
                            return_dict=True,
                        )
                    s_logits = outputs['logits'] if isinstance(outputs, dict) else getattr(outputs, 'logits', outputs[1])
                    t_logits = t_outputs['logits'] if isinstance(t_outputs, dict) else getattr(t_outputs, 'logits', t_outputs[0])
                    labels = inputs['labels']

                    # Align with CE shift
                    s_shift = s_logits[..., :-1, :].contiguous()
                    t_shift = t_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    mask = shift_labels.ne(IGNORE_INDEX)

                    if mask.any():
                        vocab = s_shift.size(-1)
                        s_flat = s_shift.view(-1, vocab)[mask.view(-1)]
                        t_flat = t_shift.view(-1, vocab)[mask.view(-1)]
                        T = float(getattr(self.args, 'kd_temperature', 1.0))
                        alpha = float(getattr(self.args, 'kd_alpha', 0.5))
                        kd = torch.nn.functional.kl_div(
                            torch.nn.functional.log_softmax(s_flat / T, dim=-1),
                            torch.nn.functional.softmax(t_flat / T, dim=-1),
                            reduction='batchmean'
                        ) * (T * T)
                        loss = loss + alpha * kd
        except Exception as e:
            print(f"KD compute error (ignored): {e}")
        
        return (loss, outputs) if return_outputs else loss

    def setup_repa_finetuning(self):
        """Setup RePaMoE fine-tuning mode with two stages"""
        print("Setting up RePaMoE fine-tuning mode...")
        
        # Check if we have RePaMoELLaVAxxxxForCausalLM
        model_class_name = self.model.__class__.__name__
        if "RePaMoE" in model_class_name:
            self.repa_state['has_repamoe'] = True
            print(f"  Detected RePaMoE model: {model_class_name}")
        else:
            raise ValueError(f"Model {model_class_name} does not support RePaMoE. "
                           "Please use RePaMoELLaVAxxxxForCausalLM.")
        
        # 1. Calculate total training steps
        num_training_steps = self.args.max_steps
        if num_training_steps <= 0:
            # Calculate from epochs and dataset size
            dataset_size = len(self.train_dataset) if self.train_dataset else 1000
            batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
            if hasattr(self.args, 'world_size'):
                batch_size *= self.args.world_size
            steps_per_epoch = max(1, dataset_size // batch_size)
            num_training_steps = steps_per_epoch * self.args.num_train_epochs
        
        self.repa_state['total_training_steps'] = num_training_steps
        
        # 2. Divide into two equal stages
        stage_1_steps = num_training_steps // 2
        stage_2_steps = num_training_steps - stage_1_steps
        
        self.repa_state['stage_1_steps'] = stage_1_steps
        self.repa_state['stage_2_steps'] = stage_2_steps
        
        # 3. Get MoE layer indices from model config
        if hasattr(self.model.config, 'moe') and 'moe_layers_idx' in self.model.config.moe:
            self.repa_state['moe_layers_idx'] = self.model.config.moe['moe_layers_idx']
        else:
            # Fallback: detect MoE layers
            self.repa_state['moe_layers_idx'] = self._detect_moe_layers()
        
        # 4. Freeze all non-MoE layers
        self._freeze_non_moe_layers()
        self._unfreeze_all_layers()
        
        # 5. Set initial gated ratio to 1.0
        self.repa_state['current_gated_ratio'] = self.repa_state['initial_gated_ratio']
        if hasattr(self.model, 'adjust_gated_ratio_all_layers'):
            self.model.adjust_gated_ratio_all_layers(self.repa_state['current_gated_ratio'])
            print(f"  Set initial gated ratio to {self.repa_state['current_gated_ratio']}")
        
        print(f"  Total training steps: {num_training_steps}")
        print(f"  Stage 1 (gated ratio reduction): {stage_1_steps} steps")
        print(f"  Stage 2 (post-reparam training): {stage_2_steps} steps")
        print(f"  Gated ratio will be reduced from {self.repa_state['initial_gated_ratio']} "
              f"to {self.repa_state['target_gated_ratio']} linearly over stage 1")
        print(f"  MoE layers: {self.repa_state['moe_layers_idx']}")
    
    def _detect_moe_layers(self):
        """Detect MoE layer indices from the model"""
        moe_layers_idx = []
        # For StableLM
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
            for i, layer in enumerate(layers):
                # Check if this layer has MoE/RePaMoE
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'deepspeed_moe'):
                    moe_layers_idx.append(i)
        # For QWen
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
            for i, layer in enumerate(layers):
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'deepspeed_moe'):
                    moe_layers_idx.append(i)
        return moe_layers_idx
    
    def _freeze_non_moe_layers(self):
        """Freeze all non-MoE layers"""
        frozen_count = 0
        unfrozen_count = 0
        
        for name, param in self.model.named_parameters():
            # Check if this parameter belongs to MoE layers
            is_moe_param = False
            for layer_idx in self.repa_state['moe_layers_idx']:
                if ((f'model.layers.{layer_idx}.mlp' in name or f'layers.{layer_idx}.mlp' in name) \
                    or (f'transformer.h.{layer_idx}.mlp' in name or f'h.{layer_idx}.mlp' in name)) \
                    and 'image_tower' not in name:
                    is_moe_param = True
                    break
            
            if is_moe_param:
                param.requires_grad = True
                unfrozen_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1
        
        print(f"  Frozen {frozen_count} non-MoE parameters, kept {unfrozen_count} MoE parameters trainable")
    
    def _unfreeze_all_layers(self):
        """Unfreeze all layers"""
        for name, param in self.model.named_parameters():
            if 'image_tower' in name or 'mm_projector' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        print(f"  Unfreeze all layers, except image_tower and mm_projector if present")

    def training_step(self, model, inputs, num_items_in_batch):
        """Override training step to handle RePaMoE logic"""
        if hasattr(self.args, 'finetune_repa_mode') and self.args.finetune_repa_mode:
            self._handle_repa_step_logic()
        return super().training_step(model, inputs, num_items_in_batch)

    def _handle_repa_step_logic(self):
        """Handle step-based logic for RePaMoE fine-tuning"""
        current_step = self.state.global_step
        
        # Determine current stage
        if current_step <= self.repa_state['stage_1_steps']:
            self.repa_state['current_stage'] = 1
            self._handle_stage_1_logic(current_step)
        else:
            # Transition to Stage 2 once
            if not self.repa_state['stage_1_complete']:
                self._transition_to_stage_2(current_step)
            self.repa_state['current_stage'] = 2
            # Stage 2: normal training, no special logic needed
    
    def _handle_stage_1_logic(self, current_step):
        """Handle Stage 1: gradually reduce gated ratio"""
        # Calculate new gated ratio based on progress through stage 1
        progress = current_step / self.repa_state['stage_1_steps']
        progress = min(1.0, progress)  # Ensure we don't exceed 1.0
        
        # Linear interpolation from initial to target ratio
        new_ratio = (self.repa_state['initial_gated_ratio'] * (1 - progress) + 
                    self.repa_state['target_gated_ratio'] * progress)
        new_ratio = round(new_ratio, 4)  # Round for cleaner logging
        
        # Update the ratio if it has changed significantly
        if abs(new_ratio - self.repa_state['current_gated_ratio']) >= 0.0001:
            self.repa_state['current_gated_ratio'] = new_ratio
            
            if hasattr(self.model, 'adjust_gated_ratio_all_layers'):
                self.model.adjust_gated_ratio_all_layers(new_ratio)
                print(f"Step {current_step}: Updated gated ratio to {new_ratio:.4f} "
                      f"(progress: {progress:.1%})")
    
    def _transition_to_stage_2(self, current_step):
        """Transition from Stage 1 to Stage 2: reparameterize and update optimizer"""
        print(f"Step {current_step}: Transitioning to Stage 2 - Reparameterization")
        
        # 1. Ensure final gated ratio is set
        if abs(self.repa_state['current_gated_ratio'] - self.repa_state['target_gated_ratio']) > 0.01:
            if hasattr(self.model, 'adjust_gated_ratio_all_layers'):
                self.model.adjust_gated_ratio_all_layers(self.repa_state['target_gated_ratio'])
                self.repa_state['current_gated_ratio'] = self.repa_state['target_gated_ratio']
                print(f"  Set final gated ratio to {self.repa_state['target_gated_ratio']}")
        
        # 2. Store old parameter IDs before reparameterization
        old_param_ids = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # Only track trainable parameters
                old_param_ids.add(id(param))
        
        # 3. Call reparameterization
        if hasattr(self.model, 'reparam_moe_layers'):
            print("  Calling reparam_moe_layers()...")
            self.model.reparam_moe_layers()
            self.repa_state['reparam_called'] = True
            print("  Reparameterization completed")
        else:
            print("  Warning: Model does not have reparam_moe_layers method")
        
        # 4. Update optimizer with new parameters
        self.model.disable_moe_allreduce()
        self._update_optimizer_after_reparam(old_param_ids)
        
        self.repa_state['stage_1_complete'] = True
        print(f"Step {current_step}: Successfully transitioned to Stage 2")
    
    def _update_optimizer_after_reparam(self, old_param_ids):
        """Update optimizer parameters after reparameterization"""
        try:
            print("  Updating optimizer after reparameterization...")
            
            # Check if this is a DeepSpeed ZeRO optimizer
            is_deepspeed_optimizer = False
            if hasattr(self.optimizer, 'state_dict'):
                try:
                    state_dict = self.optimizer.state_dict()
                    is_deepspeed_optimizer = (
                        'base_optimizer_state' in state_dict or
                        'zero_stage' in state_dict or
                        any('ep_size' in str(group.get('name', '')) for group in getattr(self.optimizer, 'param_groups', []))
                    )
                except:
                    pass
            
            if is_deepspeed_optimizer:
                print("    Detected DeepSpeed ZeRO optimizer, using specialized handling...")
                success = self._handle_deepspeed_optimizer_reparam()
                if success:
                    print("    DeepSpeed optimizer reparameterization completed successfully")
                    self.verify_optimizer_params()
                    return
                else:
                    print("    DeepSpeed handling failed, falling back to standard recreation...")
                    self._fallback_optimizer_recreation()
                    self.verify_optimizer_params()
                    return
            
            # For standard optimizers, proceed with the original approach
            print("    Using standard optimizer update approach...")
            
            # Step 1: Remove old parameter groups from optimizer
            new_param_groups = []
            removed_groups_count = 0
            
            for group in self.optimizer.param_groups:
                # Check if this group contains old parameters
                has_old_params = any(id(param) in old_param_ids for param in group['params'])
                
                if has_old_params:
                    # Filter out old parameters, keep others
                    remaining_params = [param for param in group['params'] if id(param) not in old_param_ids]
                    
                    if remaining_params:
                        # Update group with remaining parameters
                        new_group = group.copy()
                        new_group['params'] = remaining_params
                        new_param_groups.append(new_group)
                        print(f"    Updated parameter group '{group.get('name', 'unnamed')}': "
                              f"{len(group['params'])} -> {len(remaining_params)} params")
                    else:
                        # Remove empty group
                        removed_groups_count += 1
                        print(f"    Removed empty parameter group '{group.get('name', 'unnamed')}'")
                else:
                    # Keep non-affected groups as is
                    new_param_groups.append(group)
            
            # Step 2: Collect new reparameterized parameters
            new_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad and id(param) not in old_param_ids:
                    new_params.append(param)
            
            print(f"    Found {len(new_params)} new reparameterized parameters")
            
            # Step 3: Add new parameters to optimizer
            if new_params:
                # Separate by weight decay
                decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
                decay_parameters = [name for name in decay_parameters if "bias" not in name]
                
                # Get parameter names for the new parameters
                param_names_map = {id(p): n for n, p in self.model.named_parameters()}
                
                decay_params = []
                no_decay_params = []
                
                for param in new_params:
                    param_name = param_names_map.get(id(param), "repamoe_reparam_param")
                    if any(decay_name in param_name for decay_name in decay_parameters):
                        decay_params.append(param)
                    else:
                        no_decay_params.append(param)
                
                # Add new parameter groups with proper settings
                if decay_params:
                    new_group = {
                        "params": decay_params,
                        "weight_decay": self.args.weight_decay,
                        "name": "repamoe_reparam_decay",
                        "moe": True  # Mark as MoE for DeepSpeed
                    }
                    # Copy other settings from existing groups
                    if new_param_groups:
                        sample_group = new_param_groups[0]
                        for key in ['lr', 'betas', 'eps', 'momentum', 'dampening']:
                            if key in sample_group:
                                new_group[key] = sample_group[key]
                    
                    new_param_groups.append(new_group)
                    print(f"    Added {len(decay_params)} reparameterized decay params")
                
                if no_decay_params:
                    new_group = {
                        "params": no_decay_params,
                        "weight_decay": 0.0,
                        "name": "repamoe_reparam_no_decay",
                        "moe": True  # Mark as MoE for DeepSpeed
                    }
                    # Copy other settings from existing groups
                    if new_param_groups:
                        sample_group = new_param_groups[0]
                        for key in ['lr', 'betas', 'eps', 'momentum', 'dampening']:
                            if key in sample_group:
                                new_group[key] = sample_group[key]
                    
                    new_param_groups.append(new_group)
                    print(f"    Added {len(no_decay_params)} reparameterized no-decay params")
            
            # Step 4: Update optimizer with new parameter groups
            self.optimizer.param_groups = new_param_groups
            
            # Step 5: Clear optimizer state for removed parameters and initialize for new ones
            if hasattr(self.optimizer, 'state'):
                old_state_keys = list(self.optimizer.state.keys())
                for param_id in old_state_keys:
                    if param_id in old_param_ids:
                        del self.optimizer.state[param_id]
                
                # Initialize state for new parameters
                param_id = len(old_state_keys)
                for param in new_params:
                    if param_id not in self.optimizer.state:
                        self.optimizer.state[param_id] = {
                            'step': torch.tensor(0.0),
                            'exp_avg': torch.zeros_like(param.data),
                            'exp_avg_sq': torch.zeros_like(param.data)
                        }
                    param_id += 1
                
                print(f"    Cleared optimizer state for {len([k for k in old_state_keys if k in old_param_ids])} removed parameters")
                print(f"    Initialized state for {len(new_params)} new parameters")
            
            # Step 6: Verify the update
            total_params = sum(len(group['params']) for group in self.optimizer.param_groups)
            print(f"    Optimizer update completed: {len(self.optimizer.param_groups)} groups, {total_params} total parameters")
            
            # Final verification
            if not self.verify_optimizer_params():
                print("    Warning: Optimizer parameter sync verification failed after reparam")
                print("    Attempting full optimizer recreation...")
                self._fallback_optimizer_recreation()
                self.verify_optimizer_params()
            
        except Exception as e:
            print(f"    Error updating optimizer after reparam: {e}")
            print(f"    Falling back to full optimizer recreation...")
            self._fallback_optimizer_recreation()
            self.verify_optimizer_params()
    
    def verify_optimizer_params(self):
        """Verify that all trainable parameters are in the optimizer"""
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            return True
        
        # Get all trainable parameters
        trainable_params = {id(p) for p in self.model.parameters() if p.requires_grad}
        
        # Get all parameters in optimizer
        optimizer_params = set()
        for group in self.optimizer.param_groups:
            for p in group['params']:
                optimizer_params.add(id(p))
        
        # Check for missing parameters
        missing_params = trainable_params - optimizer_params
        extra_params = optimizer_params - trainable_params
        
        if missing_params:
            print(f"Warning: {len(missing_params)} trainable parameters not in optimizer!")
            # Try to identify which layers these belong to
            for name, param in self.model.named_parameters():
                if id(param) in missing_params and param.requires_grad:
                    print(f"  Missing parameter: {name}")
        
        if extra_params:
            print(f"Warning: {len(extra_params)} optimizer parameters not trainable!")
        
        if not missing_params and not extra_params:
            print(f"✓ Optimizer parameter sync verified: {len(trainable_params)} parameters")
        
        return len(missing_params) == 0
    
    def _fallback_optimizer_recreation(self):
        """Fallback method: recreate the entire optimizer with current trainable parameters"""
        try:
            print("  Recreating optimizer with all current trainable parameters...")
            
            # Save current optimizer state for recovery
            old_state_dict = None
            old_lr = None
            old_weight_decay = None
            old_betas = None
            old_eps = None
            
            if hasattr(self.optimizer, 'state_dict'):
                try:
                    old_state_dict = self.optimizer.state_dict()
                    
                    # Extract common optimizer settings from the first parameter group
                    if 'param_groups' in old_state_dict and len(old_state_dict['param_groups']) > 0:
                        first_group = old_state_dict['param_groups'][0]
                        old_lr = first_group.get('lr', self.args.learning_rate)
                        old_weight_decay = first_group.get('weight_decay', self.args.weight_decay)
                        old_betas = first_group.get('betas', (0.9, 0.999))
                        old_eps = first_group.get('eps', 1e-8)
                        
                        print(f"    Extracted optimizer settings: lr={old_lr}, weight_decay={old_weight_decay}, betas={old_betas}, eps={old_eps}")
                    
                    # For DeepSpeed ZeRO optimizer, try to extract from base_optimizer_state
                    if 'base_optimizer_state' in old_state_dict:
                        base_state = old_state_dict['base_optimizer_state']
                        if 'param_groups' in base_state and len(base_state['param_groups']) > 0:
                            first_group = base_state['param_groups'][0]
                            old_lr = first_group.get('lr', old_lr)
                            old_weight_decay = first_group.get('weight_decay', old_weight_decay)
                            old_betas = first_group.get('betas', old_betas)
                            old_eps = first_group.get('eps', old_eps)
                            
                            print(f"    Extracted from DeepSpeed base optimizer: lr={old_lr}, weight_decay={old_weight_decay}")
                            
                except Exception as e:
                    print(f"    Warning: Could not extract old optimizer state: {e}")
            
            # Clear current optimizer
            self.optimizer = None
            
            # Temporarily update args with recovered settings if available
            original_lr = self.args.learning_rate
            original_weight_decay = self.args.weight_decay
            
            if old_lr is not None:
                self.args.learning_rate = old_lr
            if old_weight_decay is not None:
                self.args.weight_decay = old_weight_decay
            
            try:
                # Recreate optimizer with current trainable parameters
                self.create_optimizer()
                
                # If we have additional settings, apply them to all parameter groups
                if old_betas is not None or old_eps is not None:
                    for group in self.optimizer.param_groups:
                        if old_betas is not None and 'betas' in group:
                            group['betas'] = old_betas
                        if old_eps is not None and 'eps' in group:
                            group['eps'] = old_eps
                
                print(f"    Successfully recreated optimizer with {len(self.optimizer.param_groups)} parameter groups")
                
                # Try to initialize momentum states for new parameters if we have step information
                if old_state_dict and 'base_optimizer_state' in old_state_dict:
                    base_state = old_state_dict['base_optimizer_state']
                    if 'state' in base_state and len(base_state['state']) > 0:
                        # Get a sample state to understand the structure
                        sample_state = next(iter(base_state['state'].values()))
                        if 'step' in sample_state:
                            step_count = sample_state['step']
                            print(f"    Initializing optimizer states with step count: {step_count}")
                            
                            # Initialize all parameter states with the current step count
                            param_id = 0
                            for group in self.optimizer.param_groups:
                                for param in group['params']:
                                    if param_id not in self.optimizer.state:
                                        self.optimizer.state[param_id] = {
                                            'step': step_count.clone() if hasattr(step_count, 'clone') else torch.tensor(float(step_count)),
                                            'exp_avg': torch.zeros_like(param.data),
                                            'exp_avg_sq': torch.zeros_like(param.data)
                                        }
                                    param_id += 1
                            
                            print(f"    Initialized momentum states for {param_id} parameters")
                
            finally:
                # Restore original args
                self.args.learning_rate = original_lr
                self.args.weight_decay = original_weight_decay
            
            print("  Optimizer recreation completed successfully")
            
        except Exception as e:
            print(f"  Failed to recreate optimizer: {e}")
            # Last resort: create with default settings
            try:
                self.optimizer = None
                self.create_optimizer()
                print("  Created optimizer with default settings as fallback")
            except Exception as e2:
                print(f"  Even fallback creation failed: {e2}")

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_no_proj_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_no_proj_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                        "name": "decay_proj_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                        "name": "no_decay_proj_parameters"
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_parameters"
                    },
                ]
            # Filter out empty parameter groups (important for RePaMoE mode where parameters are initially frozen)
            non_empty_groups = []
            for group in optimizer_grouped_parameters:
                if group.get('params') and len(group['params']) > 0:
                    non_empty_groups.append(group)
                else:
                    print(f"Skipping empty parameter group: {group.get('name', 'unnamed')}")
            optimizer_grouped_parameters = non_empty_groups
            
            # Handle MoE parameters - Apply DeepSpeed MoE grouping
            if self.args.moe_enable:
                from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
                optimizer_grouped_parameters = split_params_into_different_moe_groups_for_optimizer(optimizer_grouped_parameters)
                
                # Ensure at least one group is marked as MoE for DeepSpeed
                moe_groups_found = any(group.get('moe', False) for group in optimizer_grouped_parameters)
                
                if not moe_groups_found:
                    print("Warning: No MoE groups found after DeepSpeed grouping. Creating dedicated MoE groups...")
                    
                    # Collect MoE parameters manually
                    moe_params = []
                    non_moe_params = []
                    
                    for name, param in opt_model.named_parameters():
                        if param.requires_grad:
                            is_moe_param = False
                            # Check for various MoE parameter patterns
                            moe_patterns = ['deepspeed_moe', 'experts', 'gate', 'mlp.experts', 'mlp.gate']
                            if any(pattern in name for pattern in moe_patterns):
                                is_moe_param = True
                            
                            # For RePaMoE mode, also check MoE layer indices
                            if hasattr(self.args, 'finetune_repa_mode') and self.args.finetune_repa_mode:
                                if hasattr(self, 'repa_state') and self.repa_state.get('moe_layers_idx'):
                                    for layer_idx in self.repa_state['moe_layers_idx']:
                                        if f'model.layers.{layer_idx}.mlp' in name or f'layers.{layer_idx}.mlp' in name \
                                           or f'transformer.h.{layer_idx}.mlp' in name or f'h.{layer_idx}.mlp' in name:
                                            is_moe_param = True
                                            break
                            
                            if is_moe_param:
                                moe_params.append((name, param))
                            else:
                                non_moe_params.append((name, param))
                    
                    if moe_params:
                        # Create new parameter groups with proper MoE marking
                        new_groups = []
                        
                        # Separate MoE parameters by weight decay
                        moe_decay_params = []
                        moe_no_decay_params = []
                        
                        for name, param in moe_params:
                            print("Creating opt:", name)
                            print(param.allreduce)
                            if any(decay_name in name for decay_name in decay_parameters) and "bias" not in name:    
                                moe_decay_params.append(param)
                            else:
                                moe_no_decay_params.append(param)
                        
                        # Add MoE parameter groups
                        if moe_decay_params:
                            new_groups.append({
                                "params": moe_decay_params,
                                "weight_decay": self.args.weight_decay,
                                "name": "moe_decay_parameters",
                                "moe": True
                            })
                        
                        if moe_no_decay_params:
                            new_groups.append({
                                "params": moe_no_decay_params,
                                "weight_decay": 0.0,
                                "name": "moe_no_decay_parameters", 
                                "moe": True
                            })
                        
                        # Update existing non-MoE groups
                        for group in optimizer_grouped_parameters:
                            # Remove MoE parameters from existing groups
                            moe_param_ids = {id(p) for _, p in moe_params}
                            filtered_params = [p for p in group.get('params', []) if id(p) not in moe_param_ids]
                            
                            if filtered_params:
                                new_group = group.copy()
                                new_group['params'] = filtered_params
                                new_group['moe'] = False  # Explicitly mark as non-MoE
                                new_groups.append(new_group)
                        
                        optimizer_grouped_parameters = new_groups
                        print(f"Created {len([g for g in new_groups if g.get('moe')])} dedicated MoE parameter groups")
                        print(f"Found {len(moe_params)} MoE parameters, {len(non_moe_params)} non-MoE parameters")
                
                # Final verification that we have MoE groups
                moe_groups_final = [g for g in optimizer_grouped_parameters if g.get('moe', False)]
                if not moe_groups_final:
                    assert False, "Error: No MoE parameter groups found after final verification!"
                else:
                    print(f"✓ DeepSpeed MoE requirements satisfied: {len(moe_groups_final)} MoE groups found")
            
            # Final check to ensure we have at least one non-empty parameter group
            if not optimizer_grouped_parameters or all(len(group.get('params', [])) == 0 for group in optimizer_grouped_parameters):
                # If all parameter groups are empty, create a dummy group with at least one parameter
                # This can happen in RePaMoE mode when all parameters are initially frozen
                dummy_param = next(opt_model.parameters())
                optimizer_grouped_parameters = [{
                    "params": [dummy_param],
                    "weight_decay": 0.0,
                    "name": "dummy_param_group"
                }]
                print("Warning: All parameter groups were empty. Created dummy group to avoid optimizer error.")
            
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            #     self.optimizer = OSS(
            #         params=optimizer_grouped_parameters,
            #         optim=optimizer_cls,
            #         **optimizer_kwargs,
            #     )
            # else:
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer
    
    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    def _handle_deepspeed_optimizer_reparam(self):
        """Handle DeepSpeed ZeRO optimizer reparameterization specifically"""
        try:
            print("  Handling DeepSpeed ZeRO optimizer reparameterization...")
            
            # Extract current state information
            old_state_dict = self.optimizer.state_dict()
            
            # Extract important settings
            learning_rate = None
            weight_decay = None
            betas = None
            eps = None
            step_count = None
            
            # Try to get settings from base_optimizer_state
            if 'base_optimizer_state' in old_state_dict:
                base_state = old_state_dict['base_optimizer_state']
                if 'param_groups' in base_state and len(base_state['param_groups']) > 0:
                    first_group = base_state['param_groups'][0]
                    learning_rate = first_group.get('lr')
                    weight_decay = first_group.get('weight_decay')
                    betas = first_group.get('betas')
                    eps = first_group.get('eps')
                
                # Get step count from first parameter state
                if 'state' in base_state and len(base_state['state']) > 0:
                    first_state = next(iter(base_state['state'].values()))
                    step_count = first_state.get('step')
            
            print(f"    Extracted: lr={learning_rate}, wd={weight_decay}, betas={betas}, eps={eps}, step={step_count}")
            
            # Store original args
            original_lr = self.args.learning_rate
            original_wd = self.args.weight_decay
            
            # Update args with extracted values
            if learning_rate is not None:
                self.args.learning_rate = float(learning_rate)
            if weight_decay is not None:
                self.args.weight_decay = float(weight_decay)
            
            try:
                # Clear the optimizer
                self.optimizer = None
                
                # Recreate the optimizer
                self.create_optimizer()
                
                # Apply extracted settings to all parameter groups
                for group in self.optimizer.param_groups:
                    if learning_rate is not None:
                        group['lr'] = float(learning_rate)
                    if weight_decay is not None:
                        group['weight_decay'] = float(weight_decay)
                    if betas is not None:
                        group['betas'] = betas
                    if eps is not None:
                        group['eps'] = float(eps)
                
                print(f"    Successfully recreated DeepSpeed optimizer with {len(self.optimizer.param_groups)} groups")
                
                # Initialize optimizer states with step count if available
                if step_count is not None and hasattr(self.optimizer, 'state'):
                    param_id = 0
                    for group in self.optimizer.param_groups:
                        for param in group['params']:
                            if param_id not in self.optimizer.state:
                                self.optimizer.state[param_id] = {
                                    'step': step_count.clone() if hasattr(step_count, 'clone') else torch.tensor(float(step_count)),
                                    'exp_avg': torch.zeros_like(param.data),
                                    'exp_avg_sq': torch.zeros_like(param.data)
                                }
                            param_id += 1
                    print(f"    Initialized {param_id} parameter states with step count {step_count}")
                
            finally:
                # Restore original args
                self.args.learning_rate = original_lr
                self.args.weight_decay = original_wd
            
            return True
            
        except Exception as e:
            print(f"    Error in DeepSpeed optimizer handling: {e}")
            return False



