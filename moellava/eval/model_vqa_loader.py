import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import time
import numpy as np

from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

try:
    from fvcore.nn import FlopCountMode, flop_count
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("Warning: fvcore not available. FLOPs calculation will be disabled.")


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # 计算模型参数量
    param_stats = count_parameters(model)
    print(f"Loaded model: {model_name}")
    print_model_stats(param_stats)

    if args.return_gating_logit is not None:
        from moellava.utils import get_gating_logit_by_hook
        print(model)
        fea_hooks = get_gating_logit_by_hook(model)
        all_gating_logits = {}
    image_processor = processor['image']
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    cnt = -1
    total_generation_time = 0
    total_generated_tokens = 0
    
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        cnt += 1
        # if cnt == 30:
        #     break
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        conv = conv_templates[args.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]

        # 记录生成开始时间
        start_time = time.time()
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True if args.return_gating_logit is None else False,
                stopping_criteria=stopping_criteria
            )
        
        # 记录生成结束时间
        end_time = time.time()
        generation_time = end_time - start_time
        
        # 计算生成的token数量
        input_token_len = input_ids.shape[1]
        generated_tokens = output_ids.shape[1] - input_token_len
        
        # 更新总统计信息
        total_generation_time += generation_time
        total_generated_tokens += generated_tokens
        
        # 计算FLOPs
        if cnt == 0:  # 只在第一个样本时计算并显示FLOPs统计
            flop_stats = estimate_flops_generation(model, input_ids, image_tensor, generated_tokens)
            print_model_stats(param_stats, flop_stats, generation_time)
        elif cnt % 10 == 0:  # 每10个样本显示一次性能统计
            print(f"\nSample {cnt}: Generated {generated_tokens} tokens in {generation_time:.2f}s ({generated_tokens/generation_time:.2f} tokens/s)")
        
        if args.return_gating_logit is not None:
            # import ipdb
            # ipdb.set_trace()
            all_gating_logits[cnt] = dict(gating_logit=[i.fea for i in fea_hooks],
                                          images=image_tensor if image_tensor is None else image_tensor.detach().cpu(),
                                          input_ids=input_ids.detach().cpu(),
                                          output_ids=output_ids.detach().cpu())
            print(input_ids.shape, output_ids.shape, fea_hooks[0].fea.shape, image_tensor.shape if image_tensor is not None else [])
            # assert fea_hooks[0].fea.shape[0] + 1 == output_ids.shape[1] + 575
            print('The number of hooks is:', len(fea_hooks))

        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {
                                       "generation_time": generation_time,
                                       "generated_tokens": generated_tokens,
                                       "tokens_per_second": generated_tokens / generation_time if generation_time > 0 else 0
                                   }}) + "\n")
        # ans_file.flush()
    ans_file.close()

    # 打印总体统计信息
    if total_generated_tokens > 0:
        avg_tokens_per_second = total_generated_tokens / total_generation_time if total_generation_time > 0 else 0
        print(f"\n" + "="*60)
        print("OVERALL STATISTICS")
        print("="*60)
        print(f"Total samples processed: {cnt + 1}")
        print(f"Total tokens generated: {total_generated_tokens}")
        print(f"Total generation time: {total_generation_time:.2f} seconds")
        print(f"Average tokens per second: {avg_tokens_per_second:.2f}")
        print(f"Average tokens per sample: {total_generated_tokens / (cnt + 1):.2f}")
        print(f"Average time per sample: {total_generation_time / (cnt + 1):.2f} seconds")
        
        # 显示模型激活参数信息
        if param_stats.get('is_moe', False):
            print(f"Model Type: MoE ({param_stats['num_experts']} experts, {param_stats['experts_per_token']} active)")
            print(f"Total Parameters: {param_stats['total_params_B']:.2f}B")
            print(f"Activated Parameters: {param_stats['activated_params_B']:.2f}B ({param_stats['activated_params'] / param_stats['total_params'] * 100:.1f}% of total)")
        else:
            print(f"Model Type: Dense")
            print(f"Total Parameters: {param_stats['total_params_B']:.2f}B")
        
        print("="*60)

    if args.return_gating_logit is not None:
        torch.save(all_gating_logits, f'{args.return_gating_logit}.pt')

def count_parameters(model):
    """计算模型的参数量，包括MoE模型的激活参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算MoE相关参数
    moe_stats = analyze_moe_structure(model)
    
    result = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6,
        'total_params_B': total_params / 1e9,
        'trainable_params_B': trainable_params / 1e9
    }
    
    # 添加MoE相关统计
    if moe_stats['is_moe']:
        result.update(moe_stats)
        result['activated_params_M'] = moe_stats['activated_params'] / 1e6
        result['activated_params_B'] = moe_stats['activated_params'] / 1e9
    
    return result

def analyze_moe_structure(model):
    """分析模型的MoE结构，计算激活参数量"""
    moe_info = {
        'is_moe': False,
        'num_experts': 0,
        'experts_per_token': 0,
        'moe_layers': 0,
        'expert_params': 0,
        'non_expert_params': 0,
        'activated_params': 0
    }
    
    # 遍历模型查找MoE层
    for name, module in model.named_modules():
        # 检查是否是MoE层（根据常见的MoE命名模式）
        if any(keyword in name.lower() for keyword in ['moe', 'expert', 'mixture']):
            moe_info['is_moe'] = True
            
            # 尝试获取专家数量和选择数量
            if hasattr(module, 'num_experts'):
                moe_info['num_experts'] = max(moe_info['num_experts'], module.num_experts)
            if hasattr(module, 'top_k') or hasattr(module, 'num_experts_per_tok'):
                experts_per_token = getattr(module, 'top_k', getattr(module, 'num_experts_per_tok', 2))
                moe_info['experts_per_token'] = max(moe_info['experts_per_token'], experts_per_token)
            
            # 计算专家参数
            if 'experts' in name.lower():
                expert_params = sum(p.numel() for p in module.parameters())
                moe_info['expert_params'] += expert_params
                moe_info['moe_layers'] += 1
    
    # 如果没有直接找到MoE信息，尝试从配置中获取
    if not moe_info['is_moe'] and hasattr(model, 'config'):
        config = model.config
        if hasattr(config, 'num_experts') or hasattr(config, 'moe'):
            moe_info['is_moe'] = True
            moe_info['num_experts'] = getattr(config, 'num_experts', 8)
            moe_info['experts_per_token'] = getattr(config, 'num_experts_per_tok', 2)
            
            # 估算专家参数（基于总参数和配置）
            if hasattr(config, 'intermediate_size') and hasattr(config, 'hidden_size'):
                # 假设每个专家主要是FFN层
                expert_size = 2 * config.hidden_size * config.intermediate_size
                num_layers = getattr(config, 'num_hidden_layers', 32)
                moe_info['expert_params'] = expert_size * moe_info['num_experts'] * num_layers
                moe_info['moe_layers'] = num_layers
    
    # 计算总参数量和激活参数量
    total_model_params = sum(p.numel() for p in model.parameters())
    
    if moe_info['is_moe'] and moe_info['num_experts'] > 0 and moe_info['experts_per_token'] > 0:
        # 非专家参数 = 总参数 - 专家参数
        moe_info['non_expert_params'] = total_model_params - moe_info['expert_params']
        
        # 激活参数 = 非专家参数 + (激活专家数 / 总专家数) * 专家参数
        activation_ratio = moe_info['experts_per_token'] / moe_info['num_experts']
        activated_expert_params = moe_info['expert_params'] * activation_ratio
        moe_info['activated_params'] = moe_info['non_expert_params'] + activated_expert_params
    else:
        # 非MoE模型，激活参数等于总参数
        moe_info['activated_params'] = total_model_params
        moe_info['non_expert_params'] = total_model_params
    
    return moe_info

def estimate_flops_generation(model, input_ids, image_tensor, generated_tokens):
    """估算生成过程中的FLOPs，考虑MoE模型的激活参数"""
    try:
        # 获取模型配置
        if hasattr(model, 'config'):
            config = model.config
        elif hasattr(model, 'model') and hasattr(model.model, 'config'):
            config = model.model.config
        else:
            print("Warning: Cannot access model config for FLOPs calculation")
            return None
            
        # 基本参数
        vocab_size = getattr(config, 'vocab_size', 32000)
        hidden_size = getattr(config, 'hidden_size', 4096)
        num_layers = getattr(config, 'num_hidden_layers', 32)
        intermediate_size = getattr(config, 'intermediate_size', 11008)
        num_attention_heads = getattr(config, 'num_attention_heads', 32)
        
        # 检查是否为MoE模型
        is_moe = hasattr(config, 'num_experts') or any('moe' in name.lower() for name, _ in model.named_modules())
        num_experts = getattr(config, 'num_experts', 8) if is_moe else 1
        experts_per_token = getattr(config, 'num_experts_per_tok', 2) if is_moe else 1
        
        # 输入序列长度
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        
        # 1. Attention FLOPs: 4 * batch_size * seq_len * hidden_size^2 + 2 * batch_size * seq_len^2 * hidden_size
        attention_flops = 4 * batch_size * seq_len * hidden_size * hidden_size + 2 * batch_size * seq_len * seq_len * hidden_size
        
        # 2. FFN FLOPs for MoE vs regular models
        if is_moe:
            # MoE: 只计算激活的专家
            moe_flops = 2 * batch_size * seq_len * hidden_size * intermediate_size * experts_per_token
            # 路由网络FLOPs
            routing_flops = batch_size * seq_len * hidden_size * num_experts
            ffn_flops = moe_flops + routing_flops
        else:
            # 普通FFN
            ffn_flops = 2 * batch_size * seq_len * hidden_size * intermediate_size
        
        # 3. 每层的总FLOPs
        layer_flops = attention_flops + ffn_flops
        
        # 4. 所有层的FLOPs
        total_layer_flops = num_layers * layer_flops
        
        # 5. 输出投影FLOPs
        output_projection_flops = batch_size * seq_len * hidden_size * vocab_size
        
        # 总FLOPs（用于一次前向传播）
        total_flops_single = total_layer_flops + output_projection_flops
        
        # 生成的总FLOPs（考虑生成的每个token都需要完整的前向传播）
        total_generation_flops = total_flops_single * generated_tokens
        
        result = {
            'flops_per_token': total_flops_single,
            'total_generation_flops': total_generation_flops,
            'flops_per_token_G': total_flops_single / 1e9,
            'total_generation_flops_T': total_generation_flops / 1e12,
            'generated_tokens': generated_tokens,
            'input_length': seq_len
        }
        
        # 添加MoE相关信息
        if is_moe:
            result['is_moe'] = True
            result['num_experts'] = num_experts
            result['experts_per_token'] = experts_per_token
            result['moe_efficiency'] = experts_per_token / num_experts
            result['routing_flops'] = routing_flops * num_layers * generated_tokens
            result['routing_flops_G'] = result['routing_flops'] / 1e9
        
        return result
        
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        return None

def print_model_stats(param_stats, flop_stats=None, generation_time=None):
    """打印模型统计信息"""
    print("\n" + "="*60)
    print("MODEL STATISTICS")
    print("="*60)
    
    # 参数量统计
    print(f"Total Parameters: {param_stats['total_params']:,} ({param_stats['total_params_M']:.2f}M / {param_stats['total_params_B']:.2f}B)")
    print(f"Trainable Parameters: {param_stats['trainable_params']:,} ({param_stats['trainable_params_M']:.2f}M / {param_stats['trainable_params_B']:.2f}B)")
    
    # MoE相关统计
    if param_stats.get('is_moe', False):
        print(f"MoE Model Detected: YES")
        print(f"Number of Experts: {param_stats['num_experts']}")
        print(f"Experts per Token: {param_stats['experts_per_token']}")
        print(f"MoE Layers: {param_stats['moe_layers']}")
        print(f"Expert Parameters: {param_stats['expert_params']:,} ({param_stats['expert_params'] / 1e6:.2f}M / {param_stats['expert_params'] / 1e9:.2f}B)")
        print(f"Non-Expert Parameters: {param_stats['non_expert_params']:,} ({param_stats['non_expert_params'] / 1e6:.2f}M / {param_stats['non_expert_params'] / 1e9:.2f}B)")
        print(f"Activated Parameters: {param_stats['activated_params']:,} ({param_stats['activated_params_M']:.2f}M / {param_stats['activated_params_B']:.2f}B)")
        activation_ratio = param_stats['activated_params'] / param_stats['total_params'] * 100
        print(f"Activation Ratio: {activation_ratio:.1f}%")
    else:
        print(f"MoE Model Detected: NO")
        print(f"Activated Parameters: {param_stats['total_params']:,} ({param_stats['total_params_M']:.2f}M / {param_stats['total_params_B']:.2f}B)")
    
    # FLOPs统计
    if flop_stats:
        print(f"Input Length: {flop_stats['input_length']} tokens")
        print(f"Generated Tokens: {flop_stats['generated_tokens']} tokens")
        print(f"FLOPs per Token: {flop_stats['flops_per_token_G']:.2f} GFLOPs")
        print(f"Total Generation FLOPs: {flop_stats['total_generation_flops_T']:.2f} TFLOPs")
        
        # MoE相关FLOPs信息
        if flop_stats.get('is_moe', False):
            print(f"MoE Efficiency: {flop_stats['moe_efficiency']:.1%} (using {flop_stats['experts_per_token']}/{flop_stats['num_experts']} experts)")
            print(f"Routing FLOPs: {flop_stats['routing_flops_G']:.2f} GFLOPs")
    
    # 时间统计
    if generation_time:
        print(f"Generation Time: {generation_time:.2f} seconds")
        if flop_stats:
            tokens_per_second = flop_stats['generated_tokens'] / generation_time
            print(f"Generation Speed: {tokens_per_second:.2f} tokens/second")
            tflops_per_second = flop_stats['total_generation_flops_T'] / generation_time
            print(f"Computational Throughput: {tflops_per_second:.2f} TFLOPs/second")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--return_gating_logit", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
