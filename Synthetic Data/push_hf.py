"""
HuggingFace Hub'a dataset push etme modülü.

Kullanım:
  python push_hf.py --data-dir synthetic_outputs/text_classification_20241101_120000 --repo emredeveloper/Turkish-Synthetic-Text-Classification

Gerekli ortam değişkenleri:
  - HF_TOKEN: HuggingFace Hub token'ı
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from datasets import Dataset
    from huggingface_hub import HfApi, login, create_repo
except ImportError:
    print("Hata: Gerekli kütüphaneler eksik. Yüklemek için:")
    print("pip install datasets huggingface_hub")
    sys.exit(1)


def load_dataset_from_jsonl(jsonl_path: Path) -> Dataset:
    """JSONL dosyasından dataset yükle."""
    import json
    
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    return Dataset.from_list(data)


def push_to_hub(data_dir: Path, repo_id: str, token: Optional[str] = None) -> bool:
    """Dataset'i HuggingFace Hub'a push et."""
    
    # Token kontrolü
    if not token:
        token = os.environ.get("HF_TOKEN")
    
    if not token:
        print("Hata: HF_TOKEN ortam değişkeni tanımlı değil veya token sağlanmamış.")
        return False
    
    try:
        # Login
        login(token=token)
        api = HfApi()
        
        # Repository'yi oluştur (varsa skip)
        try:
            create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            print(f"✔️ Repository hazır: {repo_id}")
        except Exception as e:
            print(f"Repository oluşturma/kontrol hatası: {e}")
            return False
        
        # Dataset dosyalarını yükle
        train_path = data_dir / "train.jsonl"
        test_path = data_dir / "test.jsonl"
        readme_path = data_dir / "README.md"
        
        if not train_path.exists():
            print(f"Hata: {train_path} bulunamadı.")
            return False
        
        # Train dataset
        train_dataset = load_dataset_from_jsonl(train_path)
        print(f"✔️ Train dataset yüklendi: {len(train_dataset)} örnek")
        
        # Test dataset (varsa)
        test_dataset = None
        if test_path.exists():
            test_dataset = load_dataset_from_jsonl(test_path)
            print(f"✔️ Test dataset yüklendi: {len(test_dataset)} örnek")
        
        # Hub'a push
        train_dataset.push_to_hub(
            repo_id=repo_id,
            config_name="default",
            split="train"
        )
        print(f"✔️ Train split push edildi: {repo_id}")
        
        if test_dataset:
            test_dataset.push_to_hub(
                repo_id=repo_id,
                config_name="default", 
                split="test"
            )
            print(f"✔️ Test split push edildi: {repo_id}")
        
        # README dosyasını da yükle
        if readme_path.exists():
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"✔️ README.md yüklendi")
        
        print(f"🎉 Dataset başarıyla yüklendi: https://huggingface.co/datasets/{repo_id}")
        return True
        
    except Exception as e:
        print(f"Push hatası: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="HuggingFace Hub'a dataset push et")
    parser.add_argument("--data-dir", type=str, required=True, help="Veri dizini yolu")
    parser.add_argument("--repo", type=str, required=True, help="HF repo id (örn: username/dataset-name)")
    parser.add_argument("--token", type=str, help="HF token (ortam değişkeni yerine)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Hata: Veri dizini bulunamadı: {data_dir}")
        return 1
    
    success = push_to_hub(data_dir, args.repo, args.token)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())