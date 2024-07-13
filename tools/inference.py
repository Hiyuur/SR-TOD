import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
import os

# 指定模型的配置文件和 checkpoint 文件路径
#config_file = 'configs/cascade_rcnn/cascade-rcnn_r50_fpn_20e_coco.py'
#checkpoint_file = "/root/nas-public-tju/mmdet/result/cascade_rcnn/Drones/epoch_20.pth"
config_file = "/root/my_mmdet/mmdetection/configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py"
checkpoint_file = "/root/nas-public-linkdata/mmdet/result/cascade_rcnn/one_card/VisDrone/cfg_3000/epoch_12.pth"

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 初始化可视化工具
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# 从 checkpoint 中加载 Dataset_meta，并将其传递给模型的 init_detector
visualizer.dataset_meta = model.dataset_meta

# # 测试单张图片并展示结果
# #img = 'test.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
#img_path = "/root/nas-public-tju/unet/Pytorch-UNet-master/test_image/test/001172.jpg"
img_path = "/root/nas-public-linkdata/unet/Pytorch-UNet-master/test_image/test_visdrone/clean/9999945_00000_d_0000034.jpg"
#img_path = "/root/nas-public-tju/unet/Pytorch-UNet-master/test_image/test_drones/000260.jpg"
img = mmcv.imread(img_path)
result = inference_detector(model, img)

# 显示结果
img = mmcv.imread(img)
img = mmcv.imconvert(img, 'bgr', 'rgb')
#img = mmcv.imwrite

# 测试多张图片并展示结果
#img = 'test.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
# imgs_file_path = ""
# dst_dir = ""
# imgs = os.listdir(imgs_file_path)
# for img_name in imgs:
#     img_path = os.path.join(imgs_file_path,img_name)
#     name = img_name.rstrip(".jpg")
#     dir = os.path.join(dst_dir,name)
#     if not os.path.exists(dir):
#         os.makedirs(dir)

#     img = mmcv.imread(img_path)
#     result = inference_detector(model, img)

#     # 显示结果
#     img = mmcv.imread(img)
#     img = mmcv.imconvert(img, 'bgr', 'rgb')
#     #img = mmcv.imwrite


visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    show=False)

