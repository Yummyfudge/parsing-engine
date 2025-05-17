import json
import importlib
import sys
from pathlib import Path
import inspect

# Ensure project root is in sys.path for dynamic imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load runtime config
with open("config/runtime_config.json") as f:
    runtime_config = json.load(f)

SELECTED_MODEL = runtime_config["model"]
DOCUMENT_TYPE = runtime_config["document_type"]
PROFILE = runtime_config["profile"]

from layout_models import get_model_config
model_config = get_model_config(SELECTED_MODEL)

# Dynamically import layout_profiles from the selected document_type and profile
profile_module_path = f"profiles.{DOCUMENT_TYPE}.{PROFILE}.layout_profiles"
profile_module = importlib.import_module(profile_module_path)
layout_profiles = profile_module.layout_profiles
selected_profile = layout_profiles.get("default", {})

import layoutparser as lp
from PIL import Image
from pathlib import Path
import glob

# Load the layout model in a modular, backend-agnostic way
model_class = model_config["class"]
init_args = inspect.signature(model_class.__init__).parameters

kwargs = {
    "config_path": model_config["config"],
    "label_map": model_config["label_map"],
    "extra_config": ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    "model_path": model_config["model_path"],
}

if "filter_fn" in init_args:
    kwargs["filter_fn"] = selected_profile.get("filter_fn")
if "constructor_fn" in init_args:
    kwargs["constructor_fn"] = selected_profile.get("constructor_fn")

model = model_class(**kwargs)

input_dir = Path("data/input_docs")
output_dir = Path("data/output_results")
output_dir.mkdir(parents=True, exist_ok=True)

image_files = sorted(glob.glob(str(input_dir / "*.[pjPJ][pnPN]*")))  # .png, .jpg, .jpeg

for image_path_str in image_files:
    image_path = Path(image_path_str)
    print(f"🔍 Processing {image_path.name}...")

    image = Image.open(image_path)
    layout = model.detect(image)

    layout_data = []
    for i, block in enumerate(layout):
        print(f"[{i}] {block.type} at {block.block} (score: {block.score:.2f})")
        layout_data.append({
            "id": i,
            "type": block.type,
            "bbox": [block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2],
            "score": block.score,
        })

    json_output_path = output_dir / f"{image_path.stem}_layout_output.json"
    annotated_img_path = output_dir / f"{image_path.stem}_annotated.jpg"

    with open(json_output_path, "w") as f:
        json.dump(layout_data, f, indent=2)

    image_with_boxes = layout.draw(image, box_width=2, box_color="red")
    image_with_boxes.save(annotated_img_path)