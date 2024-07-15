# SR-TOD

# Introduce
  
This is the official code for the paper Visible and Clear: Finding Tiny Objects in Difference Map.

The link to the paper is https://arxiv.org/abs/2405.11276.

This project is built based on mmdetection 3.1 and mmcv 2.0.1.

**NOTE: Our paper has been accepted by ECCV 2024.**

# Environment

pytorch 1.12.0

torchvision 0.13.0

mmdetection 3.1

mmcv 2.0.1

The installation and usage of mmdetection can be referred to at the following link: https://mmdetection.readthedocs.io/en/latest/get_started.html.

To use the AI-TOD evaluation metrics, you need to download aitodpycocotools. You can install it using the following command:

```shell
pip install "git+https://github.com/jwwangchn/cocoapi-aitod.git#subdirectory=aitodpycocotools"
```

For other environment requirements, please refer to mmdetection.


# Training and Test
The training and test commands can also be referenced from mmdetection.

1 gpu:

```shell
python tools/train.py ./srtod_project/srtod_cascade_rcnn/config/srtod-cascade-rcnn_r50_fpn_1x_coco.py
```
```shell
python tools/test.py ./srtod_project/srtod_cascade_rcnn/config/srtod-cascade-rcnn_r50_fpn_1x_coco.py your_model.pth
```

If you need to use more GPUs, you should use ./tools/dist_train.sh instead of tools/train.py.

# DroneSwarms
If you want to access the DroneSwarms dataset, please visit the following link：[DroneSwarms](https://hiyuur.github.io)


