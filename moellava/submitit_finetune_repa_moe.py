import os
import argparse
import submitit
import socket
import subprocess
from dataclasses import dataclass
from typing import Optional

# This script replicates the Slurm launch in virga_finetune_repa_moe_nccl.sh
# and requeues itself every 2 hours until a completion marker is found.
# Usage:
#   python -m moellava.submitit_finetune_repa_moe \
#       --work-dir runs/moe_repa_job \
#       --output-dir finetuned_checkpoints/MoE-LLaVA-StableLM-1.6B-4e-RePa-2
# The training script is assumed to save checkpoints; we create a marker file
# when finished (you can adjust logic as needed).

@dataclass
class FinetuneJob:
    args: argparse.Namespace
    launch_cmd: Optional[str] = None

    def _build_command(self) -> str:
        a = self.args
        # Environment variables (inline) followed by python training invocation.
        # We rely on Slurm to spawn one process per GPU via submitit (tasks_per_node=gpus_per_node).
        # Training script should detect distributed env via SLURM variables.
        env_exports = {
            'WANDB_MODE': 'offline',
            'HF_DATASETS_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1',
            'NCCL_DEBUG': 'WARN',
            'NCCL_P2P_DISABLE': '1',
            'NCCL_IB_DISABLE': '1',  # adjust if IB available
            'NCCL_TREE_THRESHOLD': '0',
            'NCCL_SOCKET_IFNAME': '^docker0,lo',
            # Ensure Python can import project modules if CWD changes
            'PYTHONPATH': f"{a.project_root}:" + os.environ.get('PYTHONPATH', '')
        }
        inline_env = ' '.join(f"{k}={v}" for k,v in env_exports.items())
        # Use module invocation so path issues are avoided
        cmd = [
            "python", "-u", "-m", "moellava.train.train_mem",
            "--moe_enable", "True",
            "--num_experts", str(a.num_experts),
            "--top_k_experts", str(a.top_k_experts),
            "--capacity_factor", "1.5",
            "--moe_mode", a.moe_mode,
            "--use_residual", str(a.use_residual),
            "--router_aux_loss_coef", str(a.router_aux_loss_coef),
            "--train_modules", "gate_proj", "up_proj", "down_proj", "wg",
            "--deepspeed", a.deepspeed_config,
            "--model_name_or_path", a.model_name_or_path,
            "--version", "stablelm",
            "--data_path", a.image_tune_json, a.nlp_tune_json,
            "--image_folder", a.image_folder,
            "--image_tower", "openai/clip-vit-large-patch14-336",
            "--image_projector_type", "mlp2x_gelu",
            "--mm_vision_select_layer", "-2",
            "--mm_use_im_start_end", "False",
            "--mm_use_im_patch_token", "False",
            "--image_aspect_ratio", "pad",
            "--group_by_modality_length", "True",
            "--bf16", "True",
            "--output_dir", a.output_dir,
            "--num_train_epochs", str(a.epochs),
            "--per_device_train_batch_size", str(a.train_bs),
            "--per_device_eval_batch_size", str(a.eval_bs),
            "--gradient_accumulation_steps", str(a.grad_accum),
            "--eval_strategy", "no",
            "--save_strategy", "steps",
            "--save_steps", str(a.save_steps),
            "--save_total_limit", "1",
            "--learning_rate", str(a.lr),
            "--weight_decay", "0.",
            "--warmup_ratio", "0.03",
            "--lr_scheduler_type", "cosine",
            "--logging_steps", "1",
            "--tf32", "True",
            "--model_max_length", "2048",
            "--gradient_checkpointing", "True",
            "--dataloader_num_workers", str(a.num_workers),
            "--lazy_preprocess", "True",
            "--report_to", "tensorboard",
            "--cache_dir", a.cache_dir,
            "--report_to", "wandb",
            "--finetune_repa_mode", str(a.finetune_repa_mode),
            "--repa_gated_ratio", str(a.repa_gated_ratio)
        ]
        return f"{inline_env} {' '.join(cmd)}"

    def _modules_and_env_setup(self):
        # Load modules + activate conda env (rank-independent)
        if os.environ.get("SUBMITIT_JOB_ID"):
            # Module commands are usually functions; invoke inside a shell.
            mod_cmd = (
                "module load gcc/12.3.0 cuda/12.4.0 cudnn/9.3.0-cu12 miniconda3/23.5.2 "
                "ninja/1.11.1 sqlite/3.43.1 nccl/2.20.5-cu124 && "
                f"source activate {self.args.conda_env}"
            )
            subprocess.run(["bash", "-lc", mod_cmd], check=True)
    # Always change directory to project root to ensure relative resource paths resolve
    os.chdir(self.args.project_root)

    def __call__(self):
        from submitit.helpers import JobEnvironment
        env = JobEnvironment()
        # Map SLURM_LOCALID to LOCAL_RANK if not already
        os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
        os.makedirs(self.args.output_dir, exist_ok=True)
        marker = os.path.join(self.args.output_dir, "_done.marker")
        if os.path.exists(marker):
            print("Training already completed. Skipping.")
            return "ALREADY_DONE"
        self._modules_and_env_setup()
        if self.launch_cmd is None:
            self.launch_cmd = self._build_command()
        print(f"[submitit] Job {env.job_id} rank {env.global_rank} launching training command:\n{self.launch_cmd}")
        # Execute training command
        ret = subprocess.call(self.launch_cmd, shell=True)
        if ret == 0 and env.global_rank == 0:
            # Touch marker to indicate completion
            open(marker, 'a').close()
        if ret != 0:
            raise RuntimeError(f"Training exited with code {ret}")
        return "DONE"

    def checkpoint(self):  # Called by submitit when pre-emption/time limit/ requeue
        from submitit.helpers import DelayedSubmission
        marker = os.path.join(self.args.output_dir, "_done.marker")
        if os.path.exists(marker):
            print("[submitit] Completion marker found on checkpoint; not requeueing.")
            return None
        print("[submitit] Requeuing job (time limit or checkpoint trigger).")
        return DelayedSubmission(FinetuneJob(self.args, self.launch_cmd))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--work-dir', type=str, required=True, help='Directory for submitit logs and state.')
    p.add_argument('--output-dir', type=str, required=True, help='Training output directory (checkpoints).')
    p.add_argument('--image-tune-json', type=str, default='/scratch3/li309/data/llava_data/train_json/llava_image_tune_.json')
    p.add_argument('--nlp-tune-json', type=str, default='/scratch3/li309/data/llava_data/train_json/nlp_tune.json')
    p.add_argument('--image-folder', type=str, default='/scratch3/li309/data/llava_data/train_data')
    p.add_argument('--model-name-or-path', type=str, default='./checkpoints/MoE-LLaVA-StableLM-1.6B-4e')
    p.add_argument('--deepspeed-config', type=str, default='./scripts/zero2.json')
    p.add_argument('--moe-mode', type=str, default='sparse')
    p.add_argument('--num-experts', type=int, default=4)
    p.add_argument('--top-k-experts', type=int, default=2)
    p.add_argument('--use-residual', type=bool, default=False)
    p.add_argument('--router-aux-loss-coef', type=float, default=0.01)
    p.add_argument('--finetune-repa-mode', type=bool, default=True)
    p.add_argument('--repa-gated-ratio', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--train-bs', type=int, default=8)
    p.add_argument('--eval-bs', type=int, default=8)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--save-steps', type=int, default=4000)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--num-workers', type=int, default=18)
    p.add_argument('--cache-dir', type=str, default='./cache_dir')
    p.add_argument('--conda-env', type=str, default='/home/li309/pct_code/venv/moellava-test2')
    p.add_argument('--nodes', type=int, default=8)
    p.add_argument('--gpus-per-node', type=int, default=2)
    p.add_argument('--cpus-per-task', type=int, default=36)
    p.add_argument('--time-min', type=int, default=120, help='Time per submission (minutes). Requeue extends runtime.')
    p.add_argument('--mem-gb', type=int, default=256)
    p.add_argument('--partition', type=str, default=None, help='Optional Slurm partition')
    p.add_argument('--account', type=str, default='OD-221915', help='Slurm account (-A)')
    # Project root (default: parent directory of this file's directory)
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    p.add_argument('--project-root', type=str, default=default_root, help='Absolute path to project root (contains moellava/)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    executor = submitit.AutoExecutor(folder=args.work_dir)
    slurm_params = {}
    if args.partition:
        slurm_params['partition'] = args.partition
    if args.account:
        slurm_params['account'] = args.account
    executor.update_parameters(
        timeout_min=args.time_min,
        slurm_mem=f"{args.mem_gb}G",
        nodes=args.nodes,
        tasks_per_node=args.gpus_per_node,  # one task per GPU
        cpus_per_task=args.cpus_per_task,
        gpus_per_node=args.gpus_per_node,
        slurm_additional_parameters=slurm_params,
        name="finetune_repa_moe_nccl_submitit"
    )
    job = executor.submit(FinetuneJob(args))
    print(f"Submitted job: {job.job_id}")

if __name__ == '__main__':
    main()
