import layoutparser as lp
from PIL import Image
from pathlib import Path
import json

# 📂 Set input path
image_path = Path("data/input_docs/sample-page.png")

# Load the image
image = Image.open(image_path)

# Load the layout model
model = lp.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    model_path="/Users/jodut/Projects/parsing_engine/models/model_final.pth",
)

# Run layout detection
layout = model.detect(image)

# Collect and print layout block info
layout_data = []
for i, block in enumerate(layout):
    print(f"[{i}] {block.type} at {block.block} (score: {block.score:.2f})")
    layout_data.append({
        "id": i,
        "type": block.type,
        "bbox": [block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2],
        "score": block.score,
    })

# Save JSON output next to the image
output_path = image_path.parent / "layout_output.json"
with open(output_path, "w") as f:
    json.dump(layout_data, f, indent=2)

# Optional: Save annotated image
annotated_img_path = image_path.parent / "annotated_output.jpg"
image_with_boxes = lp.draw_box(image, layout, box_width=2, box_color="red")
image_with_boxes.save(annotated_img_path)