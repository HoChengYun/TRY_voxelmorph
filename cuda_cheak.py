import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"GPU 數量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"當前 GPU 名稱: {torch.cuda.get_device_name(0)}")