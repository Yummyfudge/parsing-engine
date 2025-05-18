import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

# Paths
image_dir = "/Users/jodut/Projects/parsing_engine/data/layout_training_sydney_notes/images"
annotation_file = "/Users/jodut/Projects/parsing_engine/scripts/coco_output/annotations.json"
output_dir = "/Users/jodut/Projects/parsing_engine/scripts/model_output"

# Register the COCO dataset
register_coco_instances("sydney_notes_train", {}, annotation_file, image_dir)

# Set up config
cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("sydney_notes_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  # Adjust if your label count changes
cfg.OUTPUT_DIR = output_dir

if __name__ == "__main__":
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()