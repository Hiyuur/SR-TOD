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
config_file = "/root/SR-TOD_mmdet/srtod_project/srtod_detectors/config/srtod-cascade-rcnn_r50-rfp_20e_coco.py"
checkpoint_file = "/root/nas-public-linkdata/mmdet/public_result/SR_TOD/srtod_detectors/Drones/epoch_20.pth"

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 初始化可视化工具
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# 从 checkpoint 中加载 Dataset_meta，并将其传递给模型的 init_detector
visualizer.dataset_meta = model.dataset_meta

# img_path = "/root/nas-public-linkdata/unet/Pytorch-UNet-master/test_image/test_visdrone/clean/9999945_00000_d_0000034.jpg"

# img = mmcv.imread(img_path)
# result = inference_detector(model, img)

# # 显示结果
# img = mmcv.imread(img)
# img = mmcv.imconvert(img, 'bgr', 'rgb')
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


# visualizer.add_datasample(
#     'result',
#     img,
#     data_sample=result,
#     draw_gt=False,
#     show=False)


# 读取视频文件
video_path = "/root/nas-public-linkdata/mmdet/public_result/src/jiqunfeixing.mp4"
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 创建文件夹用于保存结果
output_folder = "/root/nas-public-linkdata/mmdet/public_result/video/3/"
vis_folder = "/root/nas-public-linkdata/mmdet/public_result/vis/3/"
# os.makedirs(output_folder, exist_ok=True)

# 逐帧读取视频
frame_count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 保存帧为图像
    frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_filename, frame)

    img = mmcv.imread(frame_filename)
    # 使用模型进行推理
    result = inference_detector(model, frame)

    # 可视化推理结果
    result_filename = os.path.join(vis_folder, f"result_{frame_count}.jpg")

    img = mmcv.imread(img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    show=False,
    out_file=result_filename)

    frame_count += 1

# 释放视频对象
cap.release()

# 重新生成视频
images = [img for img in os.listdir(vis_folder) if img.endswith(".jpg")]
# images.sort()  # 对图像文件进行排序
images.sort(key = lambda x: int(x[7:-4])) ##文件名按数字排序
# print(images)
frame = cv2.imread(os.path.join(vis_folder, images[0]))
height, width, layers = frame.shape

video_output_path = "/root/nas-public-linkdata/mmdet/public_result/video/output_video3.mp4"
video = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(vis_folder, image)))

cv2.destroyAllWindows()
video.release()


