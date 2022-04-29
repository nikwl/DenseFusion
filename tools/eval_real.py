import _init_paths
import argparse
import os
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import trimesh
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
import cv2
from scipy.spatial import cKDTree as KDTree
import json
import utils_3d

CLASSES_FILE = 'datasets/ycb/dataset_config/classes.txt'
OBJECTS_DIR = 'models'


def load_object(object_idx):
    """
    Load an object from that object's label index
    """
    class_file = open(CLASSES_FILE)
    model_list = []
    while 1:
        class_input = class_file.readline()
        if not class_input:
            break
        model_list.append(
            class_input[:-1]
        )
    
    model_path = os.path.join(
        OBJECTS_DIR, model_list[object_idx-1], "textured.obj"
    )
    print("Loading model from: {}".format(model_path))
    return trimesh.load(model_path)


def get_bbx_from_seg(label):
    """
    Get a bounding box from a binary mask
    """

    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    img_width = 480
    img_length = 640

    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # rmax += 1
    # cmax += 1
    # r_b = rmax - rmin
    # for tt in range(len(border_list)):
    #     if r_b > border_list[tt] and r_b < border_list[tt + 1]:
    #         r_b = border_list[tt + 1]
    #         break
    # c_b = cmax - cmin
    # for tt in range(len(border_list)):
    #     if c_b > border_list[tt] and c_b < border_list[tt + 1]:
    #         c_b = border_list[tt + 1]
    #         break
    # center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    # rmin = center[0] - int(r_b / 2)
    # rmax = center[0] + int(r_b / 2)
    # cmin = center[1] - int(c_b / 2)
    # cmax = center[1] + int(c_b / 2)
    # if rmin < 0:
    #     delt = -rmin
    #     rmin = 0
    #     rmax += delt
    # if cmin < 0:
    #     delt = -cmin
    #     cmin = 0
    #     cmax += delt
    # if rmax > img_width:
    #     delt = rmax - img_width
    #     rmax = img_width
    #     rmin -= delt
    # if cmax > img_length:
    #     delt = cmax - img_length
    #     cmax = img_length
    #     cmin -= delt
    return rmin, rmax, cmin, cmax


def vis_pose(
    object_model,
    color,
    rotation, 
    translation,
):

    rotation = np.array(rotation)
    translation = np.array(translation)
    rotation = quaternion_matrix(rotation)
    translation = trimesh.transformations.translation_matrix(translation)

    # Apply pred transform
    object_model.apply_transform(
        translation @ rotation
    )

    # for x in [0, 90, 180, 270]:
    #     for y in [0, 90, 180, 270]:
    #         for z in [0, 90, 180, 270]:

    # Render
    mesh_render = utils_3d.render_mesh(
        utils_3d.normalize_unit_cube(object_model),
        bg_color=0,
        resolution=color.shape[:2][::-1],
        xrot=180,
        yrot=0,
        zrot=0,
    )
    output_img = np.hstack((
        color,
        mesh_render,
    ))

    return output_img


def depth_mask(depth, thresh=500):
    depth = cv2.morphologyEx(
        (np.logical_and(depth < thresh, depth > 5)).astype(float), 
        cv2.MORPH_OPEN, 
        np.ones((5,5)),
    )
    return cv2.dilate(depth, np.ones((5,5)), iterations=1) 
    

def clean_pointcloud(cloud, thresh=1e-1):
    dists, _ = KDTree(cloud).query(cloud, k=5)
    pts_removed_by_dist = np.vstack((
        dists[:, -1] > thresh,
    )).any(axis=0)
    cloud = np.delete(cloud, pts_removed_by_dist, axis=0)
    return cloud, pts_removed_by_dist


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default = '', help='input images')
parser.add_argument('--object_idx', type=int, default = 15, help='object index')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
opt = parser.parse_args()

# Rotation Matrix:
# 0.9999897,0.004523362,0.0002378867,-0.004526068,0.9957506,0.09197944,0.0001791805,-0.09197958,0.9957609
# Translation Vector:
# -32.0819,-2.030933,3.810172
# Intrinsic Parmeters (Cx,Cy,Fx,Fy,K1,K2,K3,K4,K5,K6,Codx,Cody,P2,P1,Notused):
# 955.6666,550.358,908.461,908.491,0.4510951,-2.503197,1.495552,0.3344906,-2.338953,1.426833,0,0,-0.000423017,0.0003900038,0
# Metric Radius, Resolution height, Resolution Width:
# 1.7,1080,1920
# DEPTH CAMERA:--
# Rotation Matrix:
# 1,0,0,0,1,0,0,0,1
# Translation Vector:
# 0,0,0
# Intrinsic Parmeters (Cx,Cy,Fx,Fy,K1,K2,K3,K4,K5,K6,Codx,Cody,P2,P1,Notused):
# 324.2682,344.5823,503.7654,503.8947,0.5065742,0.1655343,0.01092764,0.8445057,0.2703496,0.05128337,0,0,2.495428E-05,-1.414053E-05,0
# Metric Radius, Resolution height, Resolution Width:
# 1.74,576,640

assert os.path.isdir(opt.input)

# Network params
num_obj = 21
img_width = 480
img_length = 640
iteration = 2
bs = 1
num_points = 1000
num_points_mesh = 500

cam_cx = 955.6666
cam_cy = 550.358
cam_fx = 908.461
cam_fy = 908.491
cam_scale = 1000.0

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Load the 3d model
# object_model = load_object(opt.object_idx)

# Load the networks
estimator = PoseNet(num_points = num_points, num_obj = num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()

refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(opt.refine_model))
refiner.eval()

itemid = opt.object_idx

counter = 15
# img_list = []
while 1: 
    counter += 1

    try:
        img = Image.open(
            os.path.join(opt.input, "color_{}.png".format(counter))
        )
        depth = np.array(Image.open(
            os.path.join(opt.input, "depth_{}.png".format(counter))
        ))
    except FileNotFoundError:
        break

    try:
        label = np.array(Image.open(
            os.path.join(opt.input, "label_{}.png".format(counter))
        ))
    except FileNotFoundError:
        label = depth_mask(depth)
        label *= itemid

        if label.sum() == 0:
            tf = None

            json.dump(
                tf,
                open(
                    os.path.join(opt.input, "tf_{}.json".format(counter)),
                    "w",
                )
            )
            continue

    rmin, rmax, cmin, cmax = get_bbx_from_seg(label == itemid)
    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
    mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
    mask = mask_label * mask_depth
    
    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    if len(choose) > num_points:
        c_mask = np.zeros(len(choose), dtype=int)
        c_mask[:num_points] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    else:
        choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    
    dh, dw = depth.shape
    xmap = np.array([[j for i in range(dw)] for j in range(dh)])
    ymap = np.array([[i for i in range(dw)] for j in range(dh)])
    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    choose = np.array([choose])

    pt2 = depth_masked / cam_scale
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    cloud = np.concatenate((pt0, pt1, pt2), axis=1)

    # _, rmed = clean_pointcloud(cloud, thresh=5e-2)
    # choose[:, rmed] = 0

    # m = trimesh.points.PointCloud(cloud.astype(float))
    # m.export("test.ply")
    # exit()

    img_masked = np.array(img)[:, :, :3]
    img_masked = np.transpose(img_masked, (2, 0, 1))
    img_masked = img_masked[:, rmin:rmax, cmin:cmax]

    # img_masked = np.transpose(img_masked, (1, 2, 0))
    # Image.fromarray(img_masked.astype(np.uint8)).save("test.png")
    # exit(0)
    # print(cloud.min(axis=0))
    # print(cloud.max(axis=0))
    # exit()
    
    cloud = torch.from_numpy(cloud.astype(np.float32))
    choose = torch.LongTensor(choose.astype(np.int32))
    img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
    index = torch.LongTensor([itemid - 1])

    cloud = Variable(cloud).cuda()
    choose = Variable(choose).cuda()
    img_masked = Variable(img_masked).cuda()
    index = Variable(index).cuda()

    cloud = cloud.view(1, num_points, 3)
    img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])
    # print("cloud shape----------", cloud.shape)
    # print("choose shape----------", choose.shape)
    # print("img_masked shape----------", img_masked.shape)
    # print("index shape----------", index.shape)
    # exit()

    pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

    pred_c = pred_c.view(bs, num_points)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(bs * num_points, 1, 3)
    points = cloud.view(bs * num_points, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
    my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
    my_pred = np.append(my_r, my_t)

    for ite in range(0, iteration):
        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
        my_mat = quaternion_matrix(my_r)
        R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
        my_mat[0:3, 3] = my_t
        
        new_cloud = torch.bmm((cloud - T), R).contiguous()
        pred_r, pred_t = refiner(new_cloud, emb, index)
        pred_r = pred_r.view(1, 1, -1)
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
        my_r_2 = pred_r.view(-1).cpu().data.numpy()
        my_t_2 = pred_t.view(-1).cpu().data.numpy()
        my_mat_2 = quaternion_matrix(my_r_2)

        my_mat_2[0:3, 3] = my_t_2

        my_mat_final = np.dot(my_mat, my_mat_2)
        my_r_final = copy.deepcopy(my_mat_final)
        my_r_final[0:3, 3] = 0
        my_r_final = quaternion_from_matrix(my_r_final, True)
        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

        my_pred = np.append(my_r_final, my_t_final)
        my_r = my_r_final
        my_t = my_t_final

    # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

    tf = {
        "rotation" : my_r.tolist(),
        "translation" : my_t.tolist(), 
    }

    json.dump(
        tf,
        open(
            os.path.join(opt.input, "tf_{}.json".format(counter)),
            "w",
        )
    )

    # img_list.append(
    #     vis_pose(
    #         object_model,
    #         np.array(img), 
    #         my_r, 
    #         my_t,
    #     )
    # )

# utils_3d.save_gif("test.gif", img_list)