

import cv2
import os
from pathlib import Path
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

# Config paths
model_dir = "/Users/jodut/Projects/parsing_engine/scripts/model_output"
image_path = "/Users/jodut/Projects/parsing_engine/data/layout_training_sydney_notes/images/Dutson J Case 00689711 Behavioral Health Progress Note 11Sep24_page1.png"
output_path = "/Users/jodut/Projects/parsing_engine/scripts/predicted_output.png"

# Register metadata
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

# Load config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(label_names)
cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

# Run inference
image = cv2.imread(image_path)
outputs = predictor(image)

# Visualize
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("sydney_notes_train"), scale=1.0)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Save output
cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
print(f"✅ Prediction saved to: {output_path}")