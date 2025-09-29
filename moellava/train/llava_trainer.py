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
                'reparam_called': False,  # kept for compatibility, no longer used
                'initial_gated_ratio': 1.0,
                'target_gated_ratio': getattr(self.args, 'gated_ratio', 0.25),
                'total_training_steps': 0,
                'current_gated_ratio': 1.0,
                'moe_layers_idx': [],
                'has_repamoe': False,
                # New fields for two-stage LR control
                'base_lrs': None,
            }
            self.setup_repa_finetuning()

    # ---------------- Two-Stage LR helper ----------------
    def _update_two_stage_lr(self, current_step):
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            return
        if self.repa_state.get('base_lrs') is None:
            # Capture initial learning rates for all param groups
            self.repa_state['base_lrs'] = [g['lr'] for g in self.optimizer.param_groups]
        base_lrs = self.repa_state['base_lrs']
        stage_1_steps = self.repa_state['stage_1_steps']
        total_steps = self.repa_state['total_training_steps']
        # Stage 1: keep constant LR
        if current_step <= stage_1_steps:
            for g, base in zip(self.optimizer.param_groups, base_lrs):
                g['lr'] = base
        else:
            # Stage 2: linear decay from base -> 0 over remaining steps
            start = stage_1_steps
            remaining_total = max(1, total_steps - start)
            progress = min(1.0, max(0.0, (current_step - start) / remaining_total))
            scale = 1.0 - progress  # linear decay
            for g, base in zip(self.optimizer.param_groups, base_lrs):
                g['lr'] = base * scale

    def create_scheduler(self, num_training_steps: int, optimizer: Optional[torch.optim.Optimizer] = None):
        # For RePa mode we manage LR manually (constant in stage 1, decay in stage 2)
        if hasattr(self.args, 'finetune_repa_mode') and self.args.finetune_repa_mode:
            self.lr_scheduler = None
            return None
        return super().create_scheduler(num_training_steps, optimizer)

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
        # Update LR each step according to stage
        self._update_two_stage_lr(current_step)

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
        print(f"Step {current_step}: Transitioning to Stage 2 - begin LR decay (no reparameterization)")
        
        # Ensure final gated ratio applied
        if abs(self.repa_state['current_gated_ratio'] - self.repa_state['target_gated_ratio']) > 0.01:
            if hasattr(self.model, 'adjust_gated_ratio_all_layers'):
                self.model.adjust_gated_ratio_all_layers(self.repa_state['target_gated_ratio'])
                self.repa_state['current_gated_ratio'] = self.repa_state['target_gated_ratio']
                print(f"  Set final gated ratio to {self.repa_state['target_gated_ratio']}")
        
        # No reparam, no optimizer rebuild
        self.repa_state['stage_1_complete'] = True
        print(f"Step {current_step}: Entered Stage 2 (LR will now decay linearly)")

    # The old _update_optimizer_after_reparam / verify / fallback methods are retained below but unused in new flow.
    # ...existing code...
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



