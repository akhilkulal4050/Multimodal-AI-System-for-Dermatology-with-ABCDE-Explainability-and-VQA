# Install once


# Download (~14 GB, 20-40 min)
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="llava-hf/llava-v1.6-mistral-7b-hf",
    local_dir="./llava-v1.6-mistral-7b-hf",
    local_dir_use_symlinks=False,
)