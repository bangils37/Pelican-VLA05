import os
from pathlib import Path
from huggingface_hub import snapshot_download

def download_all():
    base_dir = Path("/home/anhnb9/Documents/Pelican-VLA05/pretrained_model")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    pelican_dir = base_dir / "pelican_vla05"
    cosmos_dir = base_dir / "cosmos_tokenizer"
    
    print("=" * 60)
    print("1. Downloading Cosmos Tokenizer (CI8x8)...")
    print("=" * 60)
    cosmos_path = snapshot_download(
        repo_id="nvidia/Cosmos-0.1-Tokenizer-CI8x8",
        revision="04792f8b318a26d9e54319887124a8848c3cd3ca",
        allow_patterns=["encoder.jit", "decoder.jit", "config.json", "model_config.yaml"],
        local_dir=str(cosmos_dir),
    )
    print(f"Cosmos Tokenizer downloaded to: {cosmos_path}")
    
    print("\n" + "=" * 60)
    print("2. Downloading Pelican-VLA 0.5 Pretrained Checkpoint...")
    print("=" * 60)
    pelican_path = snapshot_download(
        repo_id="X-Humanoid/Pelican-VLA05",
        local_dir=str(pelican_dir),
    )
    print(f"Pelican-VLA 0.5 Checkpoint downloaded to: {pelican_path}")
    
    print("\n" + "=" * 60)
    print("Download completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    download_all()
