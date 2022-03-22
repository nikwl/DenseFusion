import open3d as o3d
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
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor

from PIL import Image
import trimesh
import utils_3d

import numpy as np
from PIL import Image
import cv2


config = o3d.io.AzureKinectSensorConfig()
sensor = o3d.io.AzureKinectSensor(config)
if not sensor.connect(0):
    raise RuntimeError('Failed to connect to sensor')

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

testdataset = PoseDataset_linemod('eval', num_points, False, opt.dataset_root, 0.0, True)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0)

sym_list = testdataset.get_sym_list()
num_points_mesh = testdataset.get_num_points_mesh()
criterion = Loss(num_points_mesh, sym_list)
criterion_refine = Loss_refine(num_points_mesh, sym_list)

diameter = []
meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
meta = yaml.load(meta_file)
for obj in objlist:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
print(diameter)

success_count = [0 for i in range(num_objects)]
num_count = [0 for i in range(num_objects)]
fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

visualizer_list = []
visualizer_loader = {}
visualizer_inc = 0

print("Scrolling to correct object")
for i, data in enumerate(testdataloader, 0):
    points, choose, img, target, model_points, idx = data
    print(points.shape)
    print(choose.shape)
    print(img.shape)
    print(target.shape)
    print(model_points.shape)
    print(idx.shape)

    # ply_index = testdataset.list_obj[i]
    # ply_path = testdataset.object_paths[ply_index]

    # print(ply_path)
    # if ply_path != "./datasets/linemod/Linemod_preprocessed/models/obj_08.ply":
    #     continue

    print("Got through first check ")

    vis_geometry_added = False
    while True:
        rgbd = sensor.capture_frame(True)

        if rgbd is None:
            print("Skipping null frames")
            continue

        color = np.array(rgbd.color)
        depth = np.array(rgbd.depth)

        color = Image.fromarray(color).resize((640, 480))
        depth = np.array(Image.fromarray(depth).resize((640, 480)))

        points, choose, img, target, model_points, idx = testdataset.get_custom(5119, color, depth)
        # print("New data")

        points = points.unsqueeze(0)
        choose = choose.unsqueeze(0)
        img = img.unsqueeze(0)
        target = target.unsqueeze(0)
        model_points = model_points.unsqueeze(0)
        idx = idx.unsqueeze(0)

        # print(points.shape)
        # print(choose.shape)
        # print(img.shape)
        # print(target.shape)
        # print(model_points.shape)
        # print(idx.shape)
        # exit()

        if len(points.size()) == 2:
            print('No.{0} NOT Pass! Lost detection!'.format(i))
            fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))

            cv2.imshow("test", cv2.cvtColor(np.array(color), cv2.COLOR_BGR2RGB))
            if cv2.waitKey(1) == 27: 
                exit()
            continue

        input_img = img.numpy()

        points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                        Variable(choose).cuda(), \
                                                        Variable(img).cuda(), \
                                                        Variable(target).cuda(), \
                                                        Variable(model_points).cuda(), \
                                                        Variable(idx).cuda()

        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)

        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)

        for ite in range(0, iteration):
            T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
            my_mat = quaternion_matrix(my_r)
            R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            my_mat[0:3, 3] = my_t
            
            new_points = torch.bmm((points - T), R).contiguous()
            pred_r, pred_t = refiner(new_points, emb, idx)
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

        i = 5119
        model_points = model_points[0].cpu().detach().numpy()
        my_r = quaternion_matrix(my_r)[:3, :3]
        pred = np.dot(model_points, my_r.T) + my_t

        # Load the mesh
        ply_index = testdataset.list_obj[i]
        ply_path = testdataset.object_paths[ply_index]
        mesh = visualizer_loader.setdefault(
            ply_path, trimesh.load(ply_path)
        ).copy()

        # Apply pred transform
        mesh.vertices = np.dot(mesh.vertices, my_r.T) + my_t
        mesh.fix_normals()

        # Load the image
        # input_img = np.array(Image.open(testdataset.list_rgb[i]))
        input_img = np.array(color)

        # for x in [0, 90, 180, 270]:
        #     for y in [0, 90, 180, 270]:
        #         for z in [0, 90, 180, 270]:

        # Render
        mesh_render = utils_3d.render_mesh(
            utils_3d.normalize_unit_cube(mesh),
            bg_color=0,
            resolution=input_img.shape[:2][::-1],
            xrot=180,
            yrot=0,
            zrot=0,
        )
        output_img = np.hstack((
            input_img,
            mesh_render,
        ))

        cv2.imshow("test", cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) == 27: 
            exit()
    # Image.fromarray(output_img).save("test_{}_{}_{}.png".format(x, y, z)) #
    
    visualizer_list.append(output_img)

    if visualizer_inc > 100:
        break
    visualizer_inc += 1
    continue

    target = target[0].cpu().detach().numpy()

    if idx[0].item() in sym_list:
        pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
        target = torch.index_select(target, 1, inds.view(-1) - 1)
        dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
    else:
        dis = np.mean(np.linalg.norm(pred - target, axis=1))

    if dis < diameter[idx[0].item()]:
        success_count[idx[0].item()] += 1
        print('No.{0} Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
    else:
        print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
    num_count[idx[0].item()] += 1


visualizer_list = [np.expand_dims(v, axis=3) for v in visualizer_list]
fps = 15
duration = len(visualizer_list) / fps
utils_3d.save_gif(
    "test_08.gif",
    np.concatenate(visualizer_list, axis=3),
    duration=duration,
)
exit()

for i in range(num_objects):
    print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
    fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
fw.close()
