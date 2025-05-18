from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, create_model
from effdet.evaluator import CocoEvaluator
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import os
import json
from pathlib import Path
from PIL import Image
from torchvision.ops import box_convert
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import os
import json
from pathlib import Path
from PIL import Image

# Config
image_dir = Path("/Users/jodut/Projects/parsing_engine/data/layout_training_sydney_notes/images")
annotation_path = Path("/Users/jodut/Projects/parsing_engine/scripts/coco_output/annotations.json")
output_model_path = Path("/Users/jodut/Projects/parsing_engine/scripts/coco_output/layout_model_sydney.pth")

# Ensure output directory exists
output_model_path.parent.mkdir(parents=True, exist_ok=True)

# Load annotations
with open(annotation_path) as f:
    coco_json = json.load(f)

# Map category names and build id mapping
categories = coco_json["categories"]
category_map = {cat["id"]: cat["name"] for cat in categories}
num_classes = len(categories)


# Custom COCO Dataset class
class CocoLayoutDataset(Dataset):
    def __init__(self, image_dir, annotation_path, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        with open(annotation_path) as f:
            coco = json.load(f)
        self.image_map = {img["id"]: img for img in coco["images"]}
        self.annotations = coco["annotations"]
        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        # Group annotations by image_id
        self.image_to_anns = {}
        for ann in self.annotations:
            self.image_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = list(self.image_map.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_map[image_id]
        img_path = self.image_dir / image_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for ann in self.image_to_anns.get(image_id, []):
            boxes.append(ann["bbox"])
            labels.append(ann["category_id"])

        # Convert boxes to [x1, y1, x2, y2]
        boxes = torch.tensor(boxes, dtype=torch.float32)
        if boxes.numel() > 0:
            boxes = box_convert(boxes, in_fmt="xywh", out_fmt="xyxy")
        else:
            boxes = boxes.new_zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id])
        }

        if self.transform:
            image = self.transform(image)

        return image, target

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = CocoLayoutDataset(str(image_dir), str(annotation_path), transform=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Create model with explicit config
config = get_efficientdet_config('tf_efficientdet_lite0')
config.num_classes = num_classes
config.image_size = (512, 512)
config.max_det_per_image = 100
config.min_score_thresh = 0.001
model = create_model('tf_efficientdet_lite0', config, pretrained=False)
model = DetBenchTrain(model, config)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop (basic)
num_epochs = 5
for epoch in range(num_epochs):
    for images, targets in data_loader:
        images = torch.stack([img.to(device) for img in images])
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = loss_dict['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), output_model_path)
print(f"✅ Model saved to {output_model_path}")