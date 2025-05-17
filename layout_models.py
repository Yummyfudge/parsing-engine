from layoutparser.models import Detectron2LayoutModel, EfficientDetLayoutModel, PaddleDetectionLayoutModel

# Model Registry
MODEL_REGISTRY = {}

def register_model(name, config, label_map, model_path):
    def decorator(cls):
        MODEL_REGISTRY[name] = {
            "class": cls,
            "config": config,
            "label_map": label_map,
            "model_path": model_path
        }
        return cls
    return decorator

# Register Detectron2 model
@register_model(
    "detectron2",
    config="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    model_path="code_local/models/model_final.pth"
)
class Detectron2LayoutModelWrapper(Detectron2LayoutModel):
    pass

# Register EfficientDet model
@register_model(
    "efficientdet",
    config="code_local/layout_configs/efficientdet_config.yaml",
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    model_path="code_local/models/efficientdet_model.pth"
)
class EfficientDetLayoutModelWrapper(EfficientDetLayoutModel):
    pass

# Register Paddle model
@register_model(
    "paddle",
    config="lp://HJDataset/ppyolov2_r50vd_dcn_365e/config",
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    model_path=None
)
class PaddleDetectionLayoutModelWrapper(PaddleDetectionLayoutModel):
    pass

def get_model_config(model_type):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")
    return MODEL_REGISTRY[model_type]