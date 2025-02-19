from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/phi-2",
    repo_type="model",
    local_dir="./phi-2_local"
)