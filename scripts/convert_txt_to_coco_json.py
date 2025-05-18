

import os
import json
from PIL import Image
from pathlib import Path

# Paths
annotation_dir = Path("/Users/jodut/Projects/parsing_engine/data/layout_training_sydney_notes/imgAnnotations")
output_path = Path("/Users/jodut/Projects/parsing_engine/scripts/coco_output/annotations.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

# You must manually define your class names here in the correct index order
class_names = [
    "ClientProviderDemographics",
    "LeftColumnAppointment",
    "RightColumnDiagnosis",
    "VisitBody",
    "SignatureBlock",
    "FooterDocumentDatestamp",
    "PageNumber"
]

# COCO template
coco = {
    "info": {"description": "Converted from YOLO .txt"},
    "images": [],
    "annotations": [],
    "categories": []
}

# Add categories
for i, name in enumerate(class_names):
    coco["categories"].append({
        "id": i,
        "name": name,
        "supercategory": "layout"
    })

annotation_id = 1
image_id = 1

# Process all .txt files, skipping macOS dot-underscore metadata files
for txt_file in sorted(f for f in annotation_dir.glob("*.txt") if not f.name.startswith("._")):
    img_file = txt_file.with_suffix(".png")
    if not img_file.exists():
        print(f"⚠️ Missing image for {txt_file.name}")
        continue

    with Image.open(img_file) as img:
        width, height = img.size

    coco["images"].append({
        "id": image_id,
        "file_name": img_file.name,
        "width": width,
        "height": height
    })

    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x_center, y_center, box_w, box_h = map(float, parts)
            x = (x_center - box_w / 2) * width
            y = (y_center - box_h / 2) * height
            w = box_w * width
            h = box_h * height

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(cls_id),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            annotation_id += 1

    image_id += 1

# Save output
with open(output_path, "w") as out:
    json.dump(coco, out, indent=2)

print(f"✅ COCO JSON saved to: {output_path}")