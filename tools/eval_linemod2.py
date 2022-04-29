import _init_paths
import argparse
import os
import random
import numpy as np
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from PIL import Image
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
import cv2
import utils_3d


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


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

    return rmin, rmax, cmin, cmax




def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]


def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
opt = parser.parse_args()

num_objects = 13
objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
num_points = 500
iteration = 4
bs = 1
dataset_config_dir = 'datasets/linemod/dataset_config'
output_result_dir = 'experiments/eval_result/linemod'
knn = KNearestNeighbor(1)

estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator.cuda()
refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
refiner.cuda()
estimator.load_state_dict(torch.load(opt.model))
refiner.load_state_dict(torch.load(opt.refine_model))
estimator.eval()
refiner.eval()

# testdataset = PoseDataset_linemod('eval', num_points, False, opt.dataset_root, 0.0, True)
# testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

# sym_list = testdataset.get_sym_list()
# num_points_mesh = testdataset.get_num_points_mesh()
# criterion = Loss(num_points_mesh, sym_list)
# criterion_refine = Loss_refine(num_points_mesh, sym_list)

# diameter = []
# meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
# meta = yaml.load(meta_file)
# for obj in objlist:
#     diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
# print(diameter)

# success_count = [0 for i in range(num_objects)]
# num_count = [0 for i in range(num_objects)]
# fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

video_dir = "01"
input_file = open('{0}/data/{1}/test.txt'.format(opt.dataset_root, video_dir))

list_rgb = []
list_depth = []
list_label = []
tf_templates = []

while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]

    tf_templates.append('{0}/data/{1}/rgb/{2}.json'.format(opt.dataset_root, video_dir, input_line))

    list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(opt.dataset_root, video_dir, input_line))
    list_depth.append('{0}/data/{1}/depth/{2}.png'.format(opt.dataset_root, video_dir, input_line))
    list_label.append('{0}/data/{1}/mask/{2}.png'.format(opt.dataset_root, video_dir, input_line))


cam_cx = 325.26110
cam_cy = 242.04899
cam_fx = 572.41140
cam_fy = 573.57043

xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])

trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
num_pt_mesh_large = 500
num_pt_mesh_small = 500
symmetry_obj_idx = [7, 8]
cam_scale = 1.0

itemid = 1

counter = 0
while 1: 
    counter += 1

    img = Image.open(list_rgb[counter])
    depth = np.array(Image.open(list_depth[counter]))
    label = np.array(Image.open(list_label[counter]))    

    # print(label.shape)

    # # Image.fromarray(utils_3d.colorize_image(label.astype(float), cmap="gray")).save("test.png")
    # Image.fromarray(label).save("test{}.png".format(counter))
    # if counter > 10:
    #     exit()
    # continue
    # # Image.fromarray(depth_masked.astype(np.uint8)).save("test.png")
    # exit(0)

    # if region.sum() == 0:
    #     continue
    # print(depth.shape)
    # print(label.shape)
    # print(img.shape)
    # exit()

    label = np.clip(label[:, :, 0], 0, 1)

    # depth = np.expand_dims(depth, axis=2)
    # label = np.expand_dims(label, axis=2)

    rmin, rmax, cmin, cmax = get_bbx_from_seg(label)

    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
    mask_label = ma.getmaskarray(ma.masked_equal(label, 1))
    mask = mask_label * mask_depth

    # Image.fromarray(utils_3d.colorize_image(mask.astype(float), cmap="gray")).save("test.png")
    # exit()
    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    if len(choose) > num_points:
        c_mask = np.zeros(len(choose), dtype=int)
        c_mask[:num_points] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    else:
        choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    choose = np.array([choose])

    pt2 = depth_masked / cam_scale
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    cloud = np.concatenate((pt0, pt1, pt2), axis=1)

    img_masked = np.array(img)[:, :, :3]
    img_masked = np.transpose(img_masked, (2, 0, 1))
    img_masked = img_masked[:, rmin:rmax, cmin:cmax]

    # print(cloud.min(axis=0))
    # print(cloud.max(axis=0))
    # exit()

    # print(itemid)
    # if itemid != 15:
    #     exit()
    # if itemid == 15:
    #     # img_masked = np.transpose(img_masked, (1, 2, 0))
    #     Image.fromarray(utils_3d.colorize_image(mask[rmin:rmax, cmin:cmax].astype(float), cmap="gray")).save("test.png")
    #     # Image.fromarray(depth_masked.astype(np.uint8)).save("test.png")
    #     exit(0)

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
    # print("ycb cloud shape----------", cloud.shape)
    # print("ycb choose shape----------", choose.shape)
    # print("ycb img_masked shape----------", img_masked.shape)
    # print("ycb index shape----------", index.shape)
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
    # my_result_wo_refine.append(my_pred.tolist())

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

    # if itemid == 15:
    tf = {
        "rotation" : my_r.tolist(),
        "translation" : my_t.tolist(), 
    }

    json.dump(
        tf,
        open(
            tf_templates[counter],
            "w",
        )
    )
    print("Saved tf for frame ", tf_templates[counter])

    # model_points = model_points[0].cpu().detach().numpy()
    # my_r = quaternion_matrix(my_r)[:3, :3]
    # pred = np.dot(model_points, my_r.T) + my_t
    # target = target[0].cpu().detach().numpy()

    # if idx[0].item() in sym_list:
    #     pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
    #     target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
    #     inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
    #     target = torch.index_select(target, 1, inds.view(-1) - 1)
    #     dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
    # else:
    #     dis = np.mean(np.linalg.norm(pred - target, axis=1))

    # if dis < diameter[idx[0].item()]:
    #     success_count[idx[0].item()] += 1
    #     print('No.{0} Pass! Distance: {1}'.format(i, dis))
    #     fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
    # else:
    #     print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
    #     fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
    # num_count[idx[0].item()] += 1

# for i in range(num_objects):
#     print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
#     fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))
# print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
# fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
# fw.close()
