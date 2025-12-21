from huggingface_hub import HfApi, hf_hub_download
from typing import List, Dict
import json
import os

class HuggingFaceGgufFetcher:
    def __init__(self, token: str = None, purpose: str = "chat"):
        self.api = HfApi(token=token)
    
    def get_models_with_gguf(
        self, 
        model_id: str = None,
        purpose: str = "chat",
    ) -> List[Dict]:
        """
        Fetch models with GGUF files filtered by license.
        
        Args:
            licenses: List of licenses (e.g., ['mit', 'apache-2.0'])
            limit: Maximum number of models to fetch (None for all)
            sort: Sort criterion ('downloads', 'likes', 'lastModified')
            direction: Sort direction (-1 for descending, 1 for ascending)
        """
        if purpose not in ["chat", "embedding"]:
            raise ValueError("Invalid purpose. Choose either 'chat' or 'embedding'.")
        model = self.api.model_info(
            repo_id=model_id
        )

        print("Checking model:", model)

        files = self.api.list_repo_files(
            repo_id=model_id,
            repo_type="model"
            )
                    
        gguf_files = [f for f in files if f.lower().endswith('.gguf')]
        models_with_gguf = [] 
        if len(gguf_files) > 0:
            print("dzefsdgsdgsdfgdfsdgfsgsg")
            # Get detailed file info
            gguf_details = []
            for idx, gguf_file in enumerate(gguf_files):
                try:
                    file_info = self.api.get_paths_info(
                        repo_id=model.id,
                        paths=[gguf_file],
                        repo_type="model"
                    )
                    print("file_info:", file_info)
                    if file_info:
                        gguf_details.append({
                            'filename': gguf_file,
                            'size': file_info[0].size,
                            'size_formatted': self._format_size(file_info[0].size),
                            'download_url': f"https://huggingface.co/{model.id}/resolve/main/{gguf_file}",
                            'default': idx == 0
                        })
                except Exception as e:
                    gguf_details.append({
                        'filename': gguf_file,
                        'size': None,
                        'size_formatted': 'Unknown',
                        'download_url': f"https://huggingface.co/{model.id}/resolve/main/{gguf_file}",
                        'default': idx == 0
                    })
            
            # Fetch README description
            readme_description = self._get_readme_description(model_id)
            
            model_info = {
                'model_id': model_id,
                'author': model.author if hasattr(model, 'author') else model_id.split('/')[0],
                'model_name': model_id.split('/')[-1],
                'downloads': model.downloads if hasattr(model, 'downloads') else 0,
                'likes': model.likes if hasattr(model, 'likes') else 0,
                'description': readme_description,
                'purpose': purpose,
                'gguf_files': gguf_details
            }
            
            models_with_gguf.append(model_info)
            print(f"✓ Found: {model.id} ({len(gguf_details)} GGUF files)")
        else:
            print(f"✗ No GGUF files found for model: {model.id}")
            return []
        return models_with_gguf
    
    def _get_readme_description(self, repo_id: str, max_chars: int = 500) -> str:
        """Fetch README.md from the repo and return the first max_chars characters."""
        readme_path = None
        try:
            readme_path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type="model"
            )
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Strip leading whitespace and take first max_chars characters
            content = content.strip()[:max_chars]
            return content
        except Exception as e:
            print(f"Could not fetch README.md: {e}")
            return ""
        finally:
            # Clean up the downloaded file
            if readme_path and os.path.exists(readme_path):
                try:
                    os.remove(readme_path)
                except Exception:
                    pass
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human-readable format."""
        if size_bytes is None:
            return "Unknown"
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    


if __name__ == "__main__":
    fetcher = HuggingFaceGgufFetcher(token=None)
    model_id="google/gemma-3-27b-it-qat-q4_0-gguf"
    print("🚀 Starting search for models with GGUF files...")
    
    # Fetch models with GGUF files
    models = fetcher.get_models_with_gguf(
        model_id=model_id
    )
    print(json.dumps(models, indent=4))