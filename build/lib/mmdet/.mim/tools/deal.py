import cv2
import os
import numpy as np
from tqdm import tqdm

def convert_2d(r, h):
    # 矩阵减法
    #s = r - h
    h = h.astype(np.float64)
    r = r.astype(np.float64)
    s = h - r
    # if np.min(s) >= 0 and np.max(s) <= 255:
    #     #print("yes")
    #     return s
    # 线性拉伸
    # s = s - np.full(s.shape, np.min(s))
    # s = s * 255 / np.max(s)
    # s = s.astype(np.uint8)

    s = np.abs(s)
    #s = s * 255 / np.max(s) #to 255
    s = s.astype(np.uint8)
    return s

def superimposed(img,feat):
    h,w = img.shape[:2]
    new_feat = cv2.resize(feat,(w,h))
    superimposed_img = img * 0.8 + new_feat * 0.2
    
    return new_feat, superimposed_img

def get(root,src_img,dist_path):
    
    img = cv2.imread(src_img)
    up_path = os.path.join(root,"up")
    down_path = os.path.join(root,"down")
    up_files = os.listdir(up_path)
    up_feats = []
    down_feats = []
    names_up = []
    names_down = []
    for file in tqdm(up_files):
        if file == "4.jpg":
            continue

        tmp_path = os.path.join(up_path,file)
        feat = cv2.imread(tmp_path)
        new_feat, superimposed_img = superimposed(img,feat)
        
        up_feats.append(new_feat)

        tmp = file.strip(".jpg")

        names_up.append(tmp)

        name1 = "up_" + tmp + "_resize.jpg"
        name2 = "up_" + tmp + "_img.jpg"
        dist1 = os.path.join(dist_path,"up",name1)
        dist2 = os.path.join(dist_path,"up",name2)
        cv2.imwrite(dist1,new_feat)
        cv2.imwrite(dist2,superimposed_img)
    
    down_files = os.listdir(down_path)
    for file in tqdm(down_files):
        if file == "4.jpg":
            continue

        tmp_path = os.path.join(down_path,file)
        feat = cv2.imread(tmp_path)
        new_feat, superimposed_img = superimposed(img,feat)

        down_feats.append(new_feat)

        tmp = file.strip(".jpg")

        names_down.append(tmp)

        name1 = "down_" + tmp + "_resize.jpg"
        name2 = "down_" + tmp + "_img.jpg"
        dist1 = os.path.join(dist_path,"down",name1)
        dist2 = os.path.join(dist_path,"down",name2)
        cv2.imwrite(dist1,new_feat)
        cv2.imwrite(dist2,superimposed_img)

    #print(names_up)
    #print(names_down)
    for i in tqdm(range(0,len(up_feats))):
        
        if names_up[i] != names_down[i]:
            is_find = False
            for j in range(0,len(names_down)):
                if names_up[i] == names_down[j]:
                    diff_feat = convert_2d(up_feats[i],down_feats[j])
                    name1 = "diff_" + names_up[i] + "_resize.jpg"
                    name2 = "diff_" + names_up[i] + "_img.jpg"
                    superimposed_img = img * 0.8 + diff_feat * 0.2

                    dist1 = os.path.join(dist_path,"difference",name1)
                    dist2 = os.path.join(dist_path,"difference",name2)
                    cv2.imwrite(dist1,diff_feat)
                    cv2.imwrite(dist2,superimposed_img)
                    is_find = True
                        
            if not is_find:
                print("list error!")

        else:
            diff_feat = convert_2d(up_feats[i],down_feats[i])
            name1 = "diff_" + names_up[i] + "_resize.jpg"
            name2 = "diff_" + names_up[i] + "_img.jpg"
            superimposed_img = img * 0.8 + diff_feat * 0.2

            dist1 = os.path.join(dist_path,"difference",name1)
            dist2 = os.path.join(dist_path,"difference",name2)
            cv2.imwrite(dist1,diff_feat)
            cv2.imwrite(dist2,superimposed_img)

        
    



if __name__ == '__main__':
    root = "/root/nas-public-tju/mmdet/result/feature_map/cascade_rcnn/feat"
    src_img = "/root/nas-public-tju/unet/Pytorch-UNet-master/test_image/test/002269.jpg"
    dist_path = "/root/nas-public-tju/mmdet/result/feature_map/cascade_rcnn/img"
    get(root,src_img,dist_path)