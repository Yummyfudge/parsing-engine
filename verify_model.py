import torch

path = "code_local/models/efficientdet_model.pth"
model = torch.load(path, map_location="cpu")
print("✅ Model keys:", list(model.keys())[:5])