

import os
import cv2
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

# Paths
image_dir = Path("/Users/jodut/Projects/parsing_engine/data/layout_training_sydney_notes/images")
model_path = "/Users/jodut/Projects/parsing_engine/scripts/model_output/model_final.pth"
output_dir = Path("/Users/jodut/Projects/parsing_engine/scripts/predicted_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Label names
label_names = [
    "Header",
    "ClientProviderDemographics",
    "LeftColumnMetadata",
    "RightColumnNarrative",
    "VisitBody",
    "Table",
    "SignatureBlock",
    "FooterDocumentDatestamp",
    "PageNumber"
]
MetadataCatalog.get("sydney_notes_train").set(thing_classes=label_names)

# Config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(label_names)
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

# Process all images
images = sorted(f for f in image_dir.glob("*.png") if not f.name.startswith("._"))
for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Skipping unreadable image: {img_path}")
        continue

    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("sydney_notes_train"), scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    save_path = output_dir / img_path.name
    cv2.imwrite(str(save_path), out.get_image()[:, :, ::-1])
    print(f"✅ Saved: {save_path}")