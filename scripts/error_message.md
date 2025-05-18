
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.

[05/17 23:21:32 d2.data.datasets.coco]: Loaded 21 images in COCO format from /Users/jodut/Projects/parsing_engine/scripts/coco_output/annotations.json
[05/17 23:21:32 d2.data.build]: Removed 0 images with no usable annotations. 21 images left.
[05/17 23:21:32 d2.data.build]: Distribution of instances among all 9 categories:
|   category    | #instances   |   category    | #instances   |   category    | #instances   |
|:-------------:|:-------------|:-------------:|:-------------|:-------------:|:-------------|
|    Header     | 20           | ClientProvi.. | 19           | LeftColumnM.. | 19           |
| RightColumn.. | 19           |   VisitBody   | 19           |     Table     | 20           |
| SignatureBl.. | 21           | FooterDocum.. | 0            |  PageNumber   | 0            |
|               |              |               |              |               |              |
|     total     | 137          |               |              |               |              |
[05/17 23:21:32 d2.data.dataset_mapper]: [DatasetMapper] Augmentations used in training: [ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'), RandomFlip()]
[05/17 23:21:32 d2.data.build]: Using training sampler TrainingSampler
[05/17 23:21:32 d2.data.common]: Serializing the dataset using: <class 'detectron2.data.common._TorchSerializedList'>
[05/17 23:21:32 d2.data.common]: Serializing 21 elements to byte tensors and concatenating them all ...
[05/17 23:21:32 d2.data.common]: Serialized dataset takes 0.01 MiB
[05/17 23:21:32 d2.data.build]: Making batched data loader with batch_size=2
[05/17 23:21:32 d2.checkpoint.detection_checkpoint]: [DetectionCheckpointer] Loading from https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl ...
model_final_280758.pkl: 167MB [00:06, 24.9MB/s]                                                                                                     
Skip loading parameter 'roi_heads.box_predictor.cls_score.weight' to the model due to incompatible shapes: (81, 1024) in the checkpoint but (10, 1024) in the model! You might want to double check if this is expected.
Skip loading parameter 'roi_heads.box_predictor.cls_score.bias' to the model due to incompatible shapes: (81,) in the checkpoint but (10,) in the model! You might want to double check if this is expected.
Skip loading parameter 'roi_heads.box_predictor.bbox_pred.weight' to the model due to incompatible shapes: (320, 1024) in the checkpoint but (36, 1024) in the model! You might want to double check if this is expected.
Skip loading parameter 'roi_heads.box_predictor.bbox_pred.bias' to the model due to incompatible shapes: (320,) in the checkpoint but (36,) in the model! You might want to double check if this is expected.
Some model parameters or buffers are not found in the checkpoint:
roi_heads.box_predictor.bbox_pred.{bias, weight}
roi_heads.box_predictor.cls_score.{bias, weight}
[05/17 23:21:39 d2.engine.train_loop]: Starting training from iteration 0
[05/17 23:21:41 d2.engine.defaults]: Model:
GeneralizedRCNN(
  (backbone): FPN(
    (fpn_lateral2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (fpn_output2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (fpn_lateral3): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (fpn_output3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (fpn_lateral4): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
    (fpn_output4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (fpn_lateral5): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
    (fpn_output5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (top_block): LastLevelMaxPool()
    (bottom_up): ResNet(
      (stem): BasicStem(
        (conv1): Conv2d(
          3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
          (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
        )
      )
      (res2): Sequential(
        (0): BottleneckBlock(
          (shortcut): Conv2d(
            64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv1): Conv2d(
            64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv2): Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv3): Conv2d(
            64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
        )
        (1): BottleneckBlock(
          (conv1): Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv2): Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv3): Conv2d(
            64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
        )
        (2): BottleneckBlock(
          (conv1): Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv2): Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv3): Conv2d(
            64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
        )
      )
      (res3): Sequential(
        (0): BottleneckBlock(
          (shortcut): Conv2d(
            256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv1): Conv2d(
            256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv2): Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv3): Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
        )
        (1): BottleneckBlock(
          (conv1): Conv2d(
            512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv2): Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv3): Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
        )
        (2): BottleneckBlock(
          (conv1): Conv2d(
            512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv2): Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv3): Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
        )
        (3): BottleneckBlock(
          (conv1): Conv2d(
            512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv2): Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv3): Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
        )
      )
      (res4): Sequential(
        (0): BottleneckBlock(
          (shortcut): Conv2d(
            512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
          (conv1): Conv2d(
            512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (1): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (2): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (3): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (4): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (5): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
      )
      (res5): Sequential(
        (0): BottleneckBlock(
          (shortcut): Conv2d(
            1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
          )
          (conv1): Conv2d(
            1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv2): Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv3): Conv2d(
            512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
          )
        )
        (1): BottleneckBlock(
          (conv1): Conv2d(
            2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv2): Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv3): Conv2d(
            512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
          )
        )
        (2): BottleneckBlock(
          (conv1): Conv2d(
            2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv2): Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv3): Conv2d(
            512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
          )
        )
      )
    )
  )
  (proposal_generator): RPN(
    (rpn_head): StandardRPNHead(
      (conv): Conv2d(
        256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        (activation): ReLU()
      )
      (objectness_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
      (anchor_deltas): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
    )
    (anchor_generator): DefaultAnchorGenerator(
      (cell_anchors): BufferList()
    )
  )
  (roi_heads): StandardROIHeads(
    (box_pooler): ROIPooler(
      (level_poolers): ModuleList(
        (0): ROIAlign(output_size=(7, 7), spatial_scale=0.25, sampling_ratio=0, aligned=True)
        (1): ROIAlign(output_size=(7, 7), spatial_scale=0.125, sampling_ratio=0, aligned=True)
        (2): ROIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=0, aligned=True)
        (3): ROIAlign(output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=0, aligned=True)
      )
    )
    (box_head): FastRCNNConvFCHead(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (fc1): Linear(in_features=12544, out_features=1024, bias=True)
      (fc_relu1): ReLU()
      (fc2): Linear(in_features=1024, out_features=1024, bias=True)
      (fc_relu2): ReLU()
    )
    (box_predictor): FastRCNNOutputLayers(
      (cls_score): Linear(in_features=1024, out_features=10, bias=True)
      (bbox_pred): Linear(in_features=1024, out_features=36, bias=True)
    )
  )
)
WARNING [05/17 23:21:41 d2.data.datasets.coco]: 
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.

[05/17 23:21:41 d2.data.datasets.coco]: Loaded 21 images in COCO format from /Users/jodut/Projects/parsing_engine/scripts/coco_output/annotations.json
[05/17 23:21:41 d2.data.build]: Removed 0 images with no usable annotations. 21 images left.
[05/17 23:21:41 d2.data.build]: Distribution of instances among all 9 categories:
|   category    | #instances   |   category    | #instances   |   category    | #instances   |
|:-------------:|:-------------|:-------------:|:-------------|:-------------:|:-------------|
|    Header     | 20           | ClientProvi.. | 19           | LeftColumnM.. | 19           |
| RightColumn.. | 19           |   VisitBody   | 19           |     Table     | 20           |
| SignatureBl.. | 21           | FooterDocum.. | 0            |  PageNumber   | 0            |
|               |              |               |              |               |              |
|     total     | 137          |               |              |               |              |
[05/17 23:21:41 d2.data.dataset_mapper]: [DatasetMapper] Augmentations used in training: [ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'), RandomFlip()]
[05/17 23:21:41 d2.data.build]: Using training sampler TrainingSampler
[05/17 23:21:41 d2.data.common]: Serializing the dataset using: <class 'detectron2.data.common._TorchSerializedList'>
[05/17 23:21:41 d2.data.common]: Serializing 21 elements to byte tensors and concatenating them all ...
[05/17 23:21:41 d2.data.common]: Serialized dataset takes 0.01 MiB
[05/17 23:21:41 d2.data.build]: Making batched data loader with batch_size=2
[05/17 23:21:41 d2.checkpoint.detection_checkpoint]: [DetectionCheckpointer] Loading from https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl ...
[05/17 23:21:41 d2.engine.defaults]: Model:
GeneralizedRCNN(
  (backbone): FPN(
    (fpn_lateral2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (fpn_output2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (fpn_lateral3): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (fpn_output3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (fpn_lateral4): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
    (fpn_output4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (fpn_lateral5): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
    (fpn_output5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (top_block): LastLevelMaxPool()
    (bottom_up): ResNet(
      (stem): BasicStem(
        (conv1): Conv2d(
          3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
          (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
        )
      )
      (res2): Sequential(
        (0): BottleneckBlock(
          (shortcut): Conv2d(
            64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv1): Conv2d(
            64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv2): Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv3): Conv2d(
            64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
        )
        (1): BottleneckBlock(
          (conv1): Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv2): Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv3): Conv2d(
            64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
        )
        (2): BottleneckBlock(
          (conv1): Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv2): Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
          )
          (conv3): Conv2d(
            64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
        )
      )
      (res3): Sequential(
        (0): BottleneckBlock(
          (shortcut): Conv2d(
            256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv1): Conv2d(
            256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv2): Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv3): Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
        )
        (1): BottleneckBlock(
          (conv1): Conv2d(
            512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv2): Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv3): Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
        )
        (2): BottleneckBlock(
          (conv1): Conv2d(
            512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv2): Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv3): Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
        )
        (3): BottleneckBlock(
          (conv1): Conv2d(
            512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv2): Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
          )
          (conv3): Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
        )
      )
      (res4): Sequential(
        (0): BottleneckBlock(
          (shortcut): Conv2d(
            512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
          (conv1): Conv2d(
            512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (1): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (2): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (3): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (4): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
        (5): BottleneckBlock(
          (conv1): Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv2): Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
          )
          (conv3): Conv2d(
            256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
          )
        )
      )
      (res5): Sequential(
        (0): BottleneckBlock(
          (shortcut): Conv2d(
            1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
          )
          (conv1): Conv2d(
            1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv2): Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv3): Conv2d(
            512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
          )
        )
        (1): BottleneckBlock(
          (conv1): Conv2d(
            2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv2): Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv3): Conv2d(
            512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
          )
        )
        (2): BottleneckBlock(
          (conv1): Conv2d(
            2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv2): Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
          )
          (conv3): Conv2d(
            512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
          )
        )
      )
    )
  )
  (proposal_generator): RPN(
    (rpn_head): StandardRPNHead(
      (conv): Conv2d(
        256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        (activation): ReLU()
      )
      (objectness_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
      (anchor_deltas): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
    )
    (anchor_generator): DefaultAnchorGenerator(
      (cell_anchors): BufferList()
    )
  )
  (roi_heads): StandardROIHeads(
    (box_pooler): ROIPooler(
      (level_poolers): ModuleList(
        (0): ROIAlign(output_size=(7, 7), spatial_scale=0.25, sampling_ratio=0, aligned=True)
        (1): ROIAlign(output_size=(7, 7), spatial_scale=0.125, sampling_ratio=0, aligned=True)
        (2): ROIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=0, aligned=True)
        (3): ROIAlign(output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=0, aligned=True)
      )
    )
    (box_head): FastRCNNConvFCHead(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (fc1): Linear(in_features=12544, out_features=1024, bias=True)
      (fc_relu1): ReLU()
      (fc2): Linear(in_features=1024, out_features=1024, bias=True)
      (fc_relu2): ReLU()
    )
    (box_predictor): FastRCNNOutputLayers(
      (cls_score): Linear(in_features=1024, out_features=10, bias=True)
      (bbox_pred): Linear(in_features=1024, out_features=36, bias=True)
    )
  )
)
WARNING [05/17 23:21:41 d2.data.datasets.coco]: 
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.

[05/17 23:21:41 d2.data.datasets.coco]: Loaded 21 images in COCO format from /Users/jodut/Projects/parsing_engine/scripts/coco_output/annotations.json
[05/17 23:21:41 d2.data.build]: Removed 0 images with no usable annotations. 21 images left.
[05/17 23:21:41 d2.data.build]: Distribution of instances among all 9 categories:
|   category    | #instances   |   category    | #instances   |   category    | #instances   |
|:-------------:|:-------------|:-------------:|:-------------|:-------------:|:-------------|
|    Header     | 20           | ClientProvi.. | 19           | LeftColumnM.. | 19           |
| RightColumn.. | 19           |   VisitBody   | 19           |     Table     | 20           |
| SignatureBl.. | 21           | FooterDocum.. | 0            |  PageNumber   | 0            |
|               |              |               |              |               |              |
|     total     | 137          |               |              |               |              |
[05/17 23:21:41 d2.data.dataset_mapper]: [DatasetMapper] Augmentations used in training: [ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'), RandomFlip()]
[05/17 23:21:41 d2.data.build]: Using training sampler TrainingSampler
[05/17 23:21:41 d2.data.common]: Serializing the dataset using: <class 'detectron2.data.common._TorchSerializedList'>
[05/17 23:21:41 d2.data.common]: Serializing 21 elements to byte tensors and concatenating them all ...
[05/17 23:21:41 d2.data.common]: Serialized dataset takes 0.01 MiB
[05/17 23:21:41 d2.data.build]: Making batched data loader with batch_size=2
[05/17 23:21:41 d2.checkpoint.detection_checkpoint]: [DetectionCheckpointer] Loading from https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl ...
Skip loading parameter 'roi_heads.box_predictor.cls_score.weight' to the model due to incompatible shapes: (81, 1024) in the checkpoint but (10, 1024) in the model! You might want to double check if this is expected.
Skip loading parameter 'roi_heads.box_predictor.cls_score.bias' to the model due to incompatible shapes: (81,) in the checkpoint but (10,) in the model! You might want to double check if this is expected.
Skip loading parameter 'roi_heads.box_predictor.bbox_pred.weight' to the model due to incompatible shapes: (320, 1024) in the checkpoint but (36, 1024) in the model! You might want to double check if this is expected.
Skip loading parameter 'roi_heads.box_predictor.bbox_pred.bias' to the model due to incompatible shapes: (320,) in the checkpoint but (36,) in the model! You might want to double check if this is expected.
Some model parameters or buffers are not found in the checkpoint:
roi_heads.box_predictor.bbox_pred.{bias, weight}
roi_heads.box_predictor.cls_score.{bias, weight}
[05/17 23:21:41 d2.engine.train_loop]: Starting training from iteration 0
ERROR [05/17 23:21:41 d2.engine.train_loop]: Exception during training:
Traceback (most recent call last):
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/train_loop.py", line 155, in train
    self.run_step()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/defaults.py", line 530, in run_step
    self._trainer.run_step()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/train_loop.py", line 297, in run_step
    data = next(self._data_loader_iter)
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/data/common.py", line 329, in __iter__
    for d in self.dataset:
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 484, in __iter__
    return self._get_iterator()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 415, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1138, in __init__
    w.start()
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/context.py", line 284, in _Popen
    return Popen(process_obj)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/popen_spawn_posix.py", line 42, in _launch
    prep_data = spawn.get_preparation_data(process_obj._name)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/spawn.py", line 154, in get_preparation_data
    _check_not_importing_main()
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/spawn.py", line 134, in _check_not_importing_main
    raise RuntimeError('''
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
[05/17 23:21:41 d2.engine.hooks]: Total training time: 0:00:00 (0:00:00 on hooks)
[05/17 23:21:41 d2.utils.events]:  iter: 0       lr: N/A  
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/spawn.py", line 116, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/spawn.py", line 125, in _main
    prepare(preparation_data)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/spawn.py", line 236, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/spawn.py", line 287, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/runpy.py", line 288, in run_path
    return _run_module_code(code, init_globals, run_name,
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/runpy.py", line 97, in _run_module_code
    _run_code(code, mod_globals, init_globals,
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/Users/jodut/Projects/parsing_engine/scripts/train_layout_detectron2.py", line 38, in <module>
    trainer.train()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/defaults.py", line 520, in train
    super().train(self.start_iter, self.max_iter)
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/train_loop.py", line 155, in train
    self.run_step()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/defaults.py", line 530, in run_step
    self._trainer.run_step()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/train_loop.py", line 297, in run_step
    data = next(self._data_loader_iter)
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/data/common.py", line 329, in __iter__
    for d in self.dataset:
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 484, in __iter__
    return self._get_iterator()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 415, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1138, in __init__
    w.start()
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/context.py", line 284, in _Popen
    return Popen(process_obj)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/popen_spawn_posix.py", line 42, in _launch
    prep_data = spawn.get_preparation_data(process_obj._name)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/spawn.py", line 154, in get_preparation_data
    _check_not_importing_main()
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/spawn.py", line 134, in _check_not_importing_main
    raise RuntimeError('''
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
Skip loading parameter 'roi_heads.box_predictor.cls_score.weight' to the model due to incompatible shapes: (81, 1024) in the checkpoint but (10, 1024) in the model! You might want to double check if this is expected.
Skip loading parameter 'roi_heads.box_predictor.cls_score.bias' to the model due to incompatible shapes: (81,) in the checkpoint but (10,) in the model! You might want to double check if this is expected.
Skip loading parameter 'roi_heads.box_predictor.bbox_pred.weight' to the model due to incompatible shapes: (320, 1024) in the checkpoint but (36, 1024) in the model! You might want to double check if this is expected.
Skip loading parameter 'roi_heads.box_predictor.bbox_pred.bias' to the model due to incompatible shapes: (320,) in the checkpoint but (36,) in the model! You might want to double check if this is expected.
Some model parameters or buffers are not found in the checkpoint:
roi_heads.box_predictor.bbox_pred.{bias, weight}
roi_heads.box_predictor.cls_score.{bias, weight}
[05/17 23:21:41 d2.engine.train_loop]: Starting training from iteration 0
ERROR [05/17 23:21:41 d2.engine.train_loop]: Exception during training:
Traceback (most recent call last):
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/train_loop.py", line 155, in train
    self.run_step()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/defaults.py", line 530, in run_step
    self._trainer.run_step()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/train_loop.py", line 297, in run_step
    data = next(self._data_loader_iter)
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/data/common.py", line 329, in __iter__
    for d in self.dataset:
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 484, in __iter__
    return self._get_iterator()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 415, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1138, in __init__
    w.start()
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/context.py", line 284, in _Popen
    return Popen(process_obj)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/popen_spawn_posix.py", line 42, in _launch
    prep_data = spawn.get_preparation_data(process_obj._name)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/spawn.py", line 154, in get_preparation_data
    _check_not_importing_main()
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/spawn.py", line 134, in _check_not_importing_main
    raise RuntimeError('''
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
[05/17 23:21:41 d2.engine.hooks]: Total training time: 0:00:00 (0:00:00 on hooks)
[05/17 23:21:41 d2.utils.events]:  iter: 0       lr: N/A  
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/spawn.py", line 116, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/spawn.py", line 125, in _main
    prepare(preparation_data)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/spawn.py", line 236, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/spawn.py", line 287, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/runpy.py", line 288, in run_path
    return _run_module_code(code, init_globals, run_name,
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/runpy.py", line 97, in _run_module_code
    _run_code(code, mod_globals, init_globals,
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/Users/jodut/Projects/parsing_engine/scripts/train_layout_detectron2.py", line 38, in <module>
    trainer.train()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/defaults.py", line 520, in train
    super().train(self.start_iter, self.max_iter)
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/train_loop.py", line 155, in train
    self.run_step()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/defaults.py", line 530, in run_step
    self._trainer.run_step()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/train_loop.py", line 297, in run_step
    data = next(self._data_loader_iter)
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/data/common.py", line 329, in __iter__
    for d in self.dataset:
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 484, in __iter__
    return self._get_iterator()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 415, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1138, in __init__
    w.start()
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/context.py", line 284, in _Popen
    return Popen(process_obj)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/popen_spawn_posix.py", line 42, in _launch
    prep_data = spawn.get_preparation_data(process_obj._name)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/spawn.py", line 154, in get_preparation_data
    _check_not_importing_main()
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/spawn.py", line 134, in _check_not_importing_main
    raise RuntimeError('''
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
ERROR [05/17 23:21:41 d2.engine.train_loop]: Exception during training:
Traceback (most recent call last):
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1243, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/queues.py", line 113, in get
    if not self._poll(timeout):
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/connection.py", line 257, in poll
    return self._poll(timeout)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/connection.py", line 424, in _poll
    r = wait([self], timeout)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/connection.py", line 931, in wait
    ready = selector.select(timeout)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/selectors.py", line 416, in select
    fd_event_list = self._selector.poll(timeout)
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/_utils/signal_handling.py", line 73, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 16750) exited unexpectedly with exit code 1. Details are lost due to multiprocessing. Rerunning with num_workers=0 may give better error trace.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/train_loop.py", line 155, in train
    self.run_step()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/defaults.py", line 530, in run_step
    self._trainer.run_step()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/train_loop.py", line 297, in run_step
    data = next(self._data_loader_iter)
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/data/common.py", line 329, in __iter__
    for d in self.dataset:
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 701, in __next__
    data = self._next_data()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1448, in _next_data
    idx, data = self._get_data()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1412, in _get_data
    success, data = self._try_get_data()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1256, in _try_get_data
    raise RuntimeError(
RuntimeError: DataLoader worker (pid(s) 16750) exited unexpectedly
[05/17 23:21:41 d2.engine.hooks]: Total training time: 0:00:02 (0:00:00 on hooks)
[05/17 23:21:41 d2.utils.events]:  iter: 0       lr: N/A  
Traceback (most recent call last):
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1243, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/queues.py", line 113, in get
    if not self._poll(timeout):
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/connection.py", line 257, in poll
    return self._poll(timeout)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/connection.py", line 424, in _poll
    r = wait([self], timeout)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/multiprocessing/connection.py", line 931, in wait
    ready = selector.select(timeout)
  File "/Users/jodut/.pyenv/versions/3.9.18/lib/python3.9/selectors.py", line 416, in select
    fd_event_list = self._selector.poll(timeout)
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/_utils/signal_handling.py", line 73, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 16750) exited unexpectedly with exit code 1. Details are lost due to multiprocessing. Rerunning with num_workers=0 may give better error trace.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jodut/Projects/parsing_engine/scripts/train_layout_detectron2.py", line 38, in <module>
    trainer.train()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/defaults.py", line 520, in train
    super().train(self.start_iter, self.max_iter)
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/train_loop.py", line 155, in train
    self.run_step()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/defaults.py", line 530, in run_step
    self._trainer.run_step()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/engine/train_loop.py", line 297, in run_step
    data = next(self._data_loader_iter)
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/detectron2/data/common.py", line 329, in __iter__
    for d in self.dataset:
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 701, in __next__
    data = self._next_data()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1448, in _next_data
    idx, data = self._get_data()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1412, in _get_data
    success, data = self._try_get_data()
  File "/Users/jodut/Projects/parsing_engine/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1256, in _try_get_data
    raise RuntimeError(
RuntimeError: DataLoader worker (pid(s) 16750) exited unexpectedly
(.venv) jodut@Joes-MacBook-Pro parsing_engine % 