import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    # ShardedDDPOption,
    logger,
)
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

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize RePaMoE fine-tuning state
        if hasattr(self.args, 'finetune_repa_mode') and self.args.finetune_repa_mode:
            self.repa_state = {
                'moe_layers_idx': [],
                'moe_layers_num': 0,
                'n_step': 0,
                'unfrozen_layers': set(),
                'repamoe_layers': {},  # layer_idx -> RePaMoE instance
                'phase_1_steps': 0,
                'phase_2_steps': 0,
                'phase_3_steps': 0,
                'current_phase': 1,
                'steps_per_unfreeze': 0,
                'next_unfreeze_step': 0,
                'reparam_called': False
            }
            # Setup RePaMoE fine-tuning mode if enabled
            self.setup_repa_finetuning()
            
    def setup_repa_finetuning(self):
        """Setup RePaMoE fine-tuning mode"""
        
        # Import here to avoid circular imports
        if "stablelm" in self.model.config.model_type:
            from moellava.model.language_model.llava_stablelm_moe import RePaMoE
        elif "qwen" in self.model.config.model_type:
            from moellava.model.language_model.llava_qwen_moe import RePaMoE
        elif "phi" in self.model.config.model_type:
            from moellava.model.language_model.llava_phi_moe import RePaMoE
        
        # 1. Identify MoE layers
        model = self.model
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            layers = model.layers
            
        moe_layers_idx = []
        for i, layer in enumerate(layers):
            # Check if this layer has MoE (has deepspeed_moe attribute)
            if hasattr(layer.mlp, 'deepspeed_moe'):
                moe_layers_idx.append(i)
        
        self.repa_state['moe_layers_idx'] = moe_layers_idx
        self.repa_state['moe_layers_num'] = len(moe_layers_idx)
        
        # 2. Compute total training steps
        num_training_steps = self.args.max_steps
        if num_training_steps <= 0:
            # Calculate from epochs and dataset size
            dataset_size = len(self.train_dataset) if self.train_dataset else 1000
            batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
            if hasattr(self.args, 'world_size'):
                batch_size *= self.args.world_size
            steps_per_epoch = max(1, dataset_size // batch_size)
            num_training_steps = steps_per_epoch * self.args.num_train_epochs
            
        self.repa_state['n_step'] = num_training_steps
        
        # 3. Calculate phase durations
        phase_1_steps = num_training_steps // 3
        phase_2_steps = num_training_steps // 3  
        phase_3_steps = num_training_steps - phase_1_steps - phase_2_steps
        
        self.repa_state['phase_1_steps'] = phase_1_steps
        self.repa_state['phase_2_steps'] = phase_2_steps
        self.repa_state['phase_3_steps'] = phase_3_steps
        
        # 4. Calculate unfreezing schedule
        if self.repa_state['moe_layers_num'] > 0:
            steps_per_unfreeze = max(1, phase_1_steps // self.repa_state['moe_layers_num'])
            self.repa_state['steps_per_unfreeze'] = steps_per_unfreeze
            self.repa_state['next_unfreeze_step'] = steps_per_unfreeze
        
        # 5. Pre-create all RePaMoE layers but keep them frozen
        for layer_idx in moe_layers_idx:
            # Get the original MoE layer
            original_moe = layers[layer_idx].mlp
            
            # Determine the number of experts
            num_experts = 4  # Default number of experts
            if hasattr(model.config, 'moe') and 'num_experts' in model.config.moe:
                if isinstance(model.config.moe['num_experts'], list):
                    expert_idx = moe_layers_idx.index(layer_idx)
                    if expert_idx < len(model.config.moe['num_experts']):
                        num_experts = model.config.moe['num_experts'][expert_idx]
                else:
                    num_experts = model.config.moe['num_experts']
            
            # Extract expert weights from original MoE layer
            expert_weights = []
            if hasattr(original_moe, 'deepspeed_moe') and hasattr(original_moe.deepspeed_moe, 'experts'):
                experts = original_moe.deepspeed_moe.experts
                if hasattr(experts, 'deepspeed_experts'):
                    # DeepSpeed MoE structure
                    for i, expert in enumerate(experts.deepspeed_experts[:num_experts]):
                        expert_state_dict = expert.state_dict()
                        expert_weights.append(expert_state_dict)
                        print(f"  Extracted weights from expert {i} in layer {layer_idx}")
            
            # Create RePaMoE
            repamoe = RePaMoE(model.config, num_experts=num_experts)
            
            # Initialize RePaMoE experts with original MoE expert weights
            if expert_weights:
                for i, expert_weight in enumerate(expert_weights):
                    if i < len(repamoe.experts):
                        # Map the expert weights to RePaMLP structure
                        self._initialize_repamp_from_expert(repamoe.experts[i], expert_weight)
                        print(f"  Initialized RePaMoE expert {i} with original expert weights")
                
                # Initialize router weights if available
                if hasattr(original_moe, 'deepspeed_moe') and hasattr(original_moe.deepspeed_moe, 'gate'):
                    gate = original_moe.deepspeed_moe.gate
                    if hasattr(gate, 'wg') and hasattr(gate.wg, 'weight'):
                        with torch.no_grad():
                            original_router_weight = gate.wg.weight
                            # Truncate or pad router weights to match new expert count
                            if original_router_weight.shape[0] >= num_experts:
                                repamoe.router.weight.copy_(original_router_weight[:num_experts])
                            else:
                                repamoe.router.weight[:original_router_weight.shape[0]].copy_(original_router_weight)
                        print(f"  Initialized router weights from original MoE gate")
            
            # Move to the same device as the original layer
            # repamoe = repamoe.to(layers[layer_idx].mlp.device)
            
            # Call adjust_gated_ratio() to set initial gating behavior
            repamoe.adjust_gated_ratio(1.0)
            
            # Call unfreeze() to set the layer to unfrozen state
            repamoe.freeze()
            
            # Replace the MLP
            layers[layer_idx].mlp = repamoe
            
            # Store reference
            self.repa_state['repamoe_layers'][layer_idx] = repamoe
        
        # 6. Freeze all parameters initially
        for param in model.parameters():
            param.requires_grad = False
        
        # 7. Immediately unfreeze the first RePaMoE layer and subsequent layers to avoid empty optimizer
        if self.repa_state['repamoe_layers']:
            # Get the last layer (which will be unfrozen first in phase 1)
            first_layer_to_unfreeze = max(self.repa_state['moe_layers_idx'])
            if first_layer_to_unfreeze in self.repa_state['repamoe_layers']:
                # Unfreeze the RePaMoE layer itself
                repamoe = self.repa_state['repamoe_layers'][first_layer_to_unfreeze]
                repamoe.unfreeze()
                for param in repamoe.parameters():
                    param.requires_grad = True
                
                # Unfreeze all subsequent layers
                subsequent_params = self._unfreeze_subsequent_layers(first_layer_to_unfreeze)
                
                self.repa_state['unfrozen_layers'].add(first_layer_to_unfreeze)
                print(f"  Pre-unfroze RePaMoE layer {first_layer_to_unfreeze} and {len(subsequent_params)} subsequent parameters to avoid empty optimizer")
        print(f"  MoE layers: {moe_layers_idx} (total: {self.repa_state['moe_layers_num']})")
        print(f"  Total steps: {num_training_steps}")
        print(f"  Phase 1 (unfreezing): {phase_1_steps} steps")
        print(f"  Phase 2 (normal training): {phase_2_steps} steps") 
        print(f"  Phase 3 (reparam + training): {phase_3_steps} steps")
        print(f"  Steps per unfreeze: {steps_per_unfreeze}")
        print(f"  Pre-created {len(self.repa_state['repamoe_layers'])} RePaMoE layers")
    
    def training_step(self, model, inputs):
        """Override training step to handle RePaMoE logic"""
        if hasattr(self.args, 'finetune_repa_mode') and self.args.finetune_repa_mode:
            self._handle_repa_step_logic()
        return super().training_step(model, inputs)
    
    def _handle_repa_step_logic(self):
        """Handle step-based logic for RePaMoE fine-tuning"""
        current_step = self.state.global_step
        
        # Determine current phase
        if current_step < self.repa_state['phase_1_steps']:
            self.repa_state['current_phase'] = 1
            self._handle_phase_1_logic(current_step)
        elif current_step < self.repa_state['phase_1_steps'] + self.repa_state['phase_2_steps']:
            self.repa_state['current_phase'] = 2
            # Phase 2: normal training, no special logic needed
        else:
            self.repa_state['current_phase'] = 3
            self._handle_phase_3_logic(current_step)
    
    def _handle_phase_1_logic(self, current_step):
        """Handle Phase 1: gradual unfreezing of RePaMoE layers"""
        # Check if it's time to unfreeze the next layer
        if (current_step >= self.repa_state['next_unfreeze_step'] and 
            len(self.repa_state['unfrozen_layers']) < self.repa_state['moe_layers_num']):
            
            # Get the next layer to unfreeze (from last to first)
            remaining_layers = [idx for idx in self.repa_state['moe_layers_idx'] 
                              if idx not in self.repa_state['unfrozen_layers']]
            if remaining_layers:
                layer_idx = max(remaining_layers)  # unfreeze from last to first
                
                # Get the RePaMoE layer
                repamoe = self.repa_state['repamoe_layers'][layer_idx]
                repamoe.unfreeze()
                
                # Enable gradients for this layer
                newly_unfrozen_params = []
                for param in repamoe.parameters():
                    if not param.requires_grad:  # Only track newly unfrozen params
                        param.requires_grad = True
                        newly_unfrozen_params.append(param)
                
                # CRITICAL: Unfreeze all subsequent layers (including self-attn, norm layers)
                subsequent_params = self._unfreeze_subsequent_layers(layer_idx)
                newly_unfrozen_params.extend(subsequent_params)
                
                # Call adjust_gated_ratio
                gated_ratio = getattr(self.args, 'repa_gated_ratio', 1.0)
                repamoe.adjust_gated_ratio(gated_ratio)
                
                # Update state
                self.repa_state['unfrozen_layers'].add(layer_idx)
                self.repa_state['next_unfreeze_step'] += self.repa_state['steps_per_unfreeze']
                
                # CRITICAL: Add newly unfrozen parameters to optimizer
                self._add_params_to_optimizer(newly_unfrozen_params, layer_idx)
                
                # Verify optimizer parameters are synchronized
                if not self.verify_optimizer_params():
                    print("  Optimizer parameter sync failed, attempting recovery...")
                    self._fallback_optimizer_recreation()
                    self.verify_optimizer_params()
                    
                print(f"Step {current_step}: Unfroze RePaMoE layer {layer_idx} and all subsequent layers")
            else:
                # If no remaining layers, we might have pre-unfrozen the first layer
                # Just call adjust_gated_ratio for already unfrozen layers
                for layer_idx in self.repa_state['unfrozen_layers']:
                    if layer_idx in self.repa_state['repamoe_layers']:
                        repamoe = self.repa_state['repamoe_layers'][layer_idx]
                        gated_ratio = getattr(self.args, 'repa_gated_ratio', 1.0)
                        repamoe.adjust_gated_ratio(gated_ratio)
                        print(f"Step {current_step}: Adjusted gated ratio for RePaMoE layer {layer_idx}")
    
    def _handle_phase_3_logic(self, current_step):
        """Handle Phase 3: call reparam() and continue training"""
        if not self.repa_state['reparam_called']:
            print(f"Step {current_step}: Starting Phase 3 - Reparameterization")
            
            # Store old parameter IDs before reparam
            old_repamoe_param_ids = set()
            for layer_idx, repamoe in self.repa_state['repamoe_layers'].items():
                for param in repamoe.parameters():
                    old_repamoe_param_ids.add(id(param))
            
            # Call reparam() for all RePaMoE layers
            for layer_idx, repamoe in self.repa_state['repamoe_layers'].items():
                print(f"  Reparameterizing layer {layer_idx}")
                repamoe.reparam()
            
            # Update optimizer after reparameterization
            self._update_optimizer_after_reparam(old_repamoe_param_ids)
            
            self.repa_state['reparam_called'] = True
            print(f"Step {current_step}: Completed reparameterization for all RePaMoE layers")

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
            # import ipdb
            # ipdb.set_trace()
            # print('self.args.moe_enable', self.args.moe_enable)
            if self.args.moe_enable:
                from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
                optimizer_grouped_parameters = split_params_into_different_moe_groups_for_optimizer(optimizer_grouped_parameters)
            
            # Filter out empty parameter groups (important for RePaMoE mode where parameters are initially frozen)
            non_empty_groups = []
            for group in optimizer_grouped_parameters:
                if group.get('params') and len(group['params']) > 0:
                    non_empty_groups.append(group)
                else:
                    print(f"Skipping empty parameter group: {group.get('name', 'unnamed')}")
            optimizer_grouped_parameters = non_empty_groups
            
            # Handle MoE parameters
            if self.args.moe_enable:
                from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
                optimizer_grouped_parameters = split_params_into_different_moe_groups_for_optimizer(optimizer_grouped_parameters)
            
            # Handle RePaMoE mode: mark RePaMoE parameters as MoE
            if hasattr(self.args, 'finetune_repa_mode') and self.args.finetune_repa_mode:
                if hasattr(self, 'repa_state') and self.repa_state.get('repamoe_layers'):
                    # For each parameter group, check if it contains RePaMoE parameters
                    for group in optimizer_grouped_parameters:
                        # Check if any parameters in this group are from RePaMoE layers
                        has_repamoe_params = False
                        for param in group.get('params', []):
                            # Check if this parameter belongs to any RePaMoE layer
                            for layer_idx in self.repa_state['repamoe_layers']:
                                repamoe = self.repa_state['repamoe_layers'][layer_idx]
                                for repamoe_param in repamoe.parameters():
                                    if param is repamoe_param:
                                        has_repamoe_params = True
                                        break
                                if has_repamoe_params:
                                    break
                            if has_repamoe_params:
                                break
                        if has_repamoe_params:
                            group['moe'] = True
                            print(f"Marked parameter group as MoE: {group.get('name', 'unnamed')}")
            
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
    
    def _add_params_to_optimizer(self, new_params, layer_idx):
        """Add newly unfrozen parameters to the optimizer"""
        if not new_params:
            return
            
        try:
            # Separate parameters by weight decay
            decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            # Get parameter names for the new parameters
            param_names_map = {id(p): n for n, p in self.model.named_parameters()}
            
            decay_params = []
            no_decay_params = []
            
            for param in new_params:
                param_name = param_names_map.get(id(param), f"repamoe_layer_{layer_idx}_param")
                if any(decay_name in param_name for decay_name in decay_parameters):
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
            
            # Add parameter groups to optimizer
            if decay_params:
                new_group = {
                    "params": decay_params,
                    "weight_decay": self.args.weight_decay,
                    "name": f"repamoe_layer_{layer_idx}_decay",
                    "moe": True  # Mark as MoE for DeepSpeed
                }
                self.optimizer.add_param_group(new_group)
                print(f"  Added {len(decay_params)} decay params for layer {layer_idx}")
            
            if no_decay_params:
                new_group = {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                    "name": f"repamoe_layer_{layer_idx}_no_decay", 
                    "moe": True  # Mark as MoE for DeepSpeed
                }
                self.optimizer.add_param_group(new_group)
                print(f"  Added {len(no_decay_params)} no-decay params for layer {layer_idx}")
                
            # If using DeepSpeed, we need to handle it specially
            if hasattr(self, '_deepspeed_optimizer'):
                # For DeepSpeed, we might need to recreate the optimizer
                print(f"  Warning: DeepSpeed detected. New parameters may not be properly handled.")
                print(f"  Consider using a smaller learning rate schedule or recreating the optimizer.")
                
        except Exception as e:
            print(f"  Error adding parameters to optimizer: {e}")
            print(f"  Attempting fallback method...")
            self._fallback_optimizer_recreation()
            
    # Add this method to the LLaVATrainer class
    def verify_optimizer_params(self):
        """Verify that all trainable parameters are in the optimizer"""
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            return
        
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
            missing_param_objects = [p for p in self.model.parameters() if id(p) in missing_params]
            for name, param in self.model.named_parameters():
                if id(param) in missing_params:
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
            
            # Save current optimizer state
            old_state_dict = None
            if hasattr(self.optimizer, 'state_dict'):
                try:
                    old_state_dict = self.optimizer.state_dict()
                except:
                    pass
            
            # Clear current optimizer
            self.optimizer = None
            
            # Recreate optimizer with current trainable parameters
            self.create_optimizer()
            
            # Try to restore compatible state
            if old_state_dict is not None:
                try:
                    # Only restore state for parameters that still exist
                    new_state_dict = self.optimizer.state_dict()
                    for group_idx, group in enumerate(old_state_dict.get('param_groups', [])):
                        if group_idx < len(new_state_dict['param_groups']):
                            # Restore learning rate and other group settings
                            for key in ['lr', 'weight_decay', 'momentum', 'dampening']:
                                if key in group and key in new_state_dict['param_groups'][group_idx]:
                                    new_state_dict['param_groups'][group_idx][key] = group[key]
                    
                    self.optimizer.load_state_dict(new_state_dict)
                    print("  Successfully restored compatible optimizer state")
                except Exception as e:
                    print(f"  Could not restore optimizer state: {e}")
            
            print("  Optimizer recreation completed")
            
        except Exception as e:
            print(f"  Failed to recreate optimizer: {e}")
        
    def _update_optimizer_after_reparam(self, old_repamoe_param_ids):
        """Update optimizer parameters after reparameterization"""
        try:
            print("  Updating optimizer after reparameterization...")
            
            # Step 1: Remove old RePaMoE parameter groups from optimizer
            new_param_groups = []
            removed_groups_count = 0
            
            for group in self.optimizer.param_groups:
                # Check if this group contains old RePaMoE parameters
                has_old_repamoe_params = any(id(param) in old_repamoe_param_ids for param in group['params'])
                
                if has_old_repamoe_params:
                    # Filter out old RePaMoE parameters, keep others
                    remaining_params = [param for param in group['params'] if id(param) not in old_repamoe_param_ids]
                    
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
                    # Keep non-RePaMoE groups as is
                    new_param_groups.append(group)
            
            # Step 2: Collect new reparameterized parameters
            new_repamoe_params = []
            for layer_idx, repamoe in self.repa_state['repamoe_layers'].items():
                for param in repamoe.parameters():
                    if param.requires_grad:
                        new_repamoe_params.append(param)
            
            print(f"    Found {len(new_repamoe_params)} new reparameterized parameters")
            
            # Step 3: Add new reparameterized parameters to optimizer
            if new_repamoe_params:
                # Separate by weight decay
                decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
                decay_parameters = [name for name in decay_parameters if "bias" not in name]
                
                # Get parameter names for the new parameters
                param_names_map = {id(p): n for n, p in self.model.named_parameters()}
                
                decay_params = []
                no_decay_params = []
                
                for param in new_repamoe_params:
                    param_name = param_names_map.get(id(param), "repamoe_reparam_param")
                    if any(decay_name in param_name for decay_name in decay_parameters):
                        decay_params.append(param)
                    else:
                        no_decay_params.append(param)
                
                # Add new parameter groups
                if decay_params:
                    new_group = {
                        "params": decay_params,
                        "weight_decay": self.args.weight_decay,
                        "name": "repamoe_reparam_decay",
                        "moe": True  # Mark as MoE for DeepSpeed
                    }
                    new_param_groups.append(new_group)
                    print(f"    Added {len(decay_params)} reparameterized decay params")
                
                if no_decay_params:
                    new_group = {
                        "params": no_decay_params,
                        "weight_decay": 0.0,
                        "name": "repamoe_reparam_no_decay",
                        "moe": True  # Mark as MoE for DeepSpeed
                    }
                    new_param_groups.append(new_group)
                    print(f"    Added {len(no_decay_params)} reparameterized no-decay params")
            
            # Step 4: Update optimizer with new parameter groups
            self.optimizer.param_groups = new_param_groups
            
            # Step 5: Clear optimizer state for removed parameters
            if hasattr(self.optimizer, 'state'):
                old_state_keys = list(self.optimizer.state.keys())
                for param_id in old_state_keys:
                    if param_id in old_repamoe_param_ids:
                        del self.optimizer.state[param_id]
                print(f"    Cleared optimizer state for {len([k for k in old_state_keys if k in old_repamoe_param_ids])} removed parameters")
            
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
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    def _initialize_repamp_from_expert(self, repamp, expert_state_dict):
        """Initialize RePaMLP from original expert weights"""
        try:
            # Map the expert state dict keys to RePaMLP structure
            # Typical expert structure: gate_proj, up_proj, down_proj
            key_mapping = {
                'gate_proj.weight': 'gate_proj.weight',
                'up_proj.weight': 'up_proj.weight', 
                'down_proj.weight': 'down_proj.weight',
                'gate_proj.bias': 'gate_proj.bias',
                'up_proj.bias': 'up_proj.bias',
                'down_proj.bias': 'down_proj.bias'
            }
            
            # Load weights that exist in both expert and RePaMLP
            with torch.no_grad():
                for expert_key, repamp_key in key_mapping.items():
                    if expert_key in expert_state_dict:
                        if hasattr(repamp, repamp_key.split('.')[0]):
                            layer = getattr(repamp, repamp_key.split('.')[0])
                            if hasattr(layer, repamp_key.split('.')[1]):
                                param = getattr(layer, repamp_key.split('.')[1])
                                if param is not None:
                                    expert_param = expert_state_dict[expert_key]
                                    if param.shape == expert_param.shape:
                                        param.copy_(expert_param)
                                    else:
                                        print(f"    Warning: Shape mismatch for {repamp_key}: "
                                              f"RePaMLP {param.shape} vs Expert {expert_param.shape}")
                                        
        except Exception as e:
            print(f"    Warning: Failed to initialize RePaMLP from expert weights: {e}")
            print(f"    Available expert keys: {list(expert_state_dict.keys())}")
    
    def _unfreeze_subsequent_layers(self, moe_layer_idx):
        """
        Unfreeze all layers after the given MoE layer index.
        This includes self_attn, norm layers, and all subsequent decoder layers.
        """
        newly_unfrozen_params = []
        
        try:
            # Access the language model layers
            if hasattr(self.model.model, 'layers'):
                layers = self.model.model.layers
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                layers = self.model.model.layers
            else:
                print(f"  Warning: Could not find language model layers for subsequent unfreezing")
                return newly_unfrozen_params
            
            # Unfreeze all layers from moe_layer_idx onwards
            for layer_idx in range(moe_layer_idx, len(layers)):
                layer = layers[layer_idx]
                
                # For the current MoE layer, unfreeze non-MoE components (self_attn, norms)
                if layer_idx == moe_layer_idx:
                    # Unfreeze self_attn
                    if hasattr(layer, 'self_attn'):
                        for param in layer.self_attn.parameters():
                            if not param.requires_grad:
                                param.requires_grad = True
                                newly_unfrozen_params.append(param)
                    
                    # Unfreeze norm layers
                    if hasattr(layer, 'input_layernorm'):
                        for param in layer.input_layernorm.parameters():
                            if not param.requires_grad:
                                param.requires_grad = True
                                newly_unfrozen_params.append(param)
                    
                    if hasattr(layer, 'post_attention_layernorm'):
                        for param in layer.post_attention_layernorm.parameters():
                            if not param.requires_grad:
                                param.requires_grad = True
                                newly_unfrozen_params.append(param)
                
                # For subsequent layers, unfreeze everything
                elif layer_idx > moe_layer_idx:
                    for param in layer.parameters():
                        if not param.requires_grad:
                            param.requires_grad = True
                            newly_unfrozen_params.append(param)
            
            # Also unfreeze final norm and lm_head if present
            if hasattr(self.model.model, 'norm'):
                for param in self.model.model.norm.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        newly_unfrozen_params.append(param)
            
            if hasattr(self.model, 'lm_head'):
                for param in self.model.lm_head.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        newly_unfrozen_params.append(param)
            
            print(f"  Unfroze {len(newly_unfrozen_params)} subsequent parameters after layer {moe_layer_idx}")
            
        except Exception as e:
            print(f"  Error unfreezing subsequent layers: {e}")
            print(f"  Continuing with just MoE layer parameters...")
        
        return newly_unfrozen_params



