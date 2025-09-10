from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="LanguageBind/MoE-LLaVA-StableLM-1.6B-4e",
    local_dir="./checkpoints/MoE-LLaVA-StableLM-1.6B-4e",
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="LanguageBind/MoE-LLaVA-Qwen-1.8B-4e",
    local_dir="./checkpoints/MoE-LLaVA-Qwen-1.8B-4e",
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="LanguageBind/MoE-LLaVA-Phi2-2.7B-4e",
    local_dir="./checkpoints/MoE-LLaVA-Phi2-2.7B-4e",
    local_dir_use_symlinks=False
)