

from layoutparser.models import Detectron2LayoutModel, EfficientDetLayoutModel, PaddleDetectionLayoutModel

def get_model_config(model_type):
    if model_type == "detectron2":
        return {
            "class": Detectron2LayoutModel,
            "config": "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            "label_map": {
                0: "Text",
                1: "Title",
                2: "List",
                3: "Table",
                4: "Figure"
            },
            "model_path": "code_local/models/model_final.pth"
        }

    elif model_type == "efficientdet":
        return {
            "class": EfficientDetLayoutModel,
            "config": "lp://PrimaLayout/mobilenetv2_coco/config",
            "label_map": {
                0: "Text",
                1: "Title",
                2: "List",
                3: "Table",
                4: "Figure"
            },
            "model_path": None
        }

    elif model_type == "paddle":
        return {
            "class": PaddleDetectionLayoutModel,
            "config": "lp://HJDataset/ppyolov2_r50vd_dcn_365e/config",
            "label_map": {
                0: "Text",
                1: "Title",
                2: "List",
                3: "Table",
                4: "Figure"
            },
            "model_path": None
        }

    raise ValueError(f"Unsupported model type: {model_type}")