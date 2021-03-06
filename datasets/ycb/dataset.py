import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
from lib.transformations import quaternion_from_euler, euler_matrix, random_quaternion, quaternion_matrix
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import trimesh
import tqdm

class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine):
        if mode == 'train':
            self.path = root + '/dataset_config/train_data_list_subset_half.txt'
        elif mode == 'test':
            self.path = root + '/dataset_config/test_data_list.txt'
        self.num_pt = num_pt
        self.root = root
        self.add_noise = add_noise
        self.noise_trans = noise_trans

        self._cache = {}
        self._do_caching = True

        self.list = []
        self.real = []
        self.syn = []
        input_file = open(self.path)
        print("reading file list...")
        idx = 0
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            if input_line[:5] == 'data/':
                self.real.append(idx)
            else:
                self.syn.append(idx)
            self.list.append(input_line)
            idx += 1
        input_file.close()

        self.length = len(self.list)
        self.len_real = len(self.real)
        self.len_syn = len(self.syn)

        # >>> Load the original classes
        class_file = open('datasets/ycb/dataset_config/classes.txt')
        class_mapping = {}
        class_id = 1
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            class_mapping[class_input] = class_id
            
            class_id += 1

        # >>> Load the class subset
        class_file = open('datasets/ycb/dataset_config/classes_subset.txt')
        self.cld = {}
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            
            # Retreive the correct id
            class_id = class_mapping[class_input]
            
            # Load the points
            input_file = '{0}/models/{1}/points.xyz'.format(self.root, class_input[:-1])
            self.cld[class_id] = self.load(input_file)

            # self.cld[class_id] = []
            # while 1:
            #     input_line = input_file.readline()
            #     if not input_line:
            #         break
            #     input_line = input_line[:-1].split(' ')
            #     self.cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            # self.cld[class_id] = np.array(self.cld[class_id])
            # input_file.close()

        self._num_objects = len(self.cld)

        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.symmetry_obj_idx = [12, 15, 18, 19, 20]
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.refine = refine
        self.front_num = 2

        # self[len(self)-1]
        # print("Pruning objects from the dataset...")
        # self._remapped_getitem = None
        # print("Using {} objects".format(self._num_objects))
        # self.prune()
        if mode == "train":
            self.load_npy()

    @property
    def num_objects(self):
        return self._num_objects

    def load(self, f_in):
        """
        An autocaching loader
        """
        if f_in not in self._cache:
            ext = os.path.splitext(f_in)[-1]
            if ext == ".png":
                data = Image.open(f_in)
            elif ext == ".mat":
                data = scio.loadmat(f_in)
            elif ext == ".xyz":
                data = np.loadtxt(f_in, delimiter=" ")
            elif ext == ".ply":
                data = trimesh.load(f_in).vertices.copy()
            else:
                raise RuntimeError("Unknown extension: {}".format(ext))
            if not self._do_caching:
                return data
            self._cache[f_in] = data
        
        return self._cache[f_in]
    
    def prune(self):
        """
        Build a remapped list of objects
        """
        self._do_caching = False
        remapped_getitem = []
        pbar = tqdm.tqdm(range(len(self)))
        for i in pbar:
            try:
                data = self[i]
            except FileNotFoundError:
                pbar.write("FileNotFoundError: {}".format(i))
                continue
            if data is not None:
                remapped_getitem.append(i)
        self._remapped_getitem = remapped_getitem
        self._do_caching = True
        print("Retained {} / {} samples".format(len(self), self.length))

    def load_npy(self):
        print("Loading numpy data ...")
        self.list_rgb = np.load(self.root + "/list_rgb.npy")
        print("list_rgb: {}".format(self.list_rgb.shape[-1]))
        print("list: {}".format(len(self.list)))
        # assert len(self.list) == self.list_rgb.shape[-1]
        self.list_depth = np.load(self.root + "/list_depth.npy")
        self.list_label = np.load(self.root + "/list_label.npy")
        self.list_meta = np.load(self.root + "/list_meta.npy", allow_pickle=True)
        self.length = self.list_rgb.shape[-1]

    def get_object(self, name):
        """
        Return the points associated with a given model
        """
        return self.load('{0}/models/{1}/points.xyz'.format(self.root, name))

    def __getitem__(self, index):
        # >>> Remap getitem
        # if self._remapped_getitem is not None:
        #     index = self._remapped_getitem[index]

        if hasattr(self, "list_rgb"):
            img = self.list_rgb[:, :, :, index]
            img = Image.fromarray(img)
            depth = self.list_depth[:, :, index]
            label = self.list_label[:, :, index]
            meta = self.list_meta[index]
        else:
            print("Loading from disk...")
            try:
                img = self.load('{0}/{1}-color.png'.format(self.root, self.list[index]))
                depth = np.array(self.load('{0}/{1}-depth.png'.format(self.root, self.list[index])))
                label = np.array(self.load('{0}/{1}-label.png'.format(self.root, self.list[index])))
                meta = self.load('{0}/{1}-meta.mat'.format(self.root, self.list[index]))
            except FileNotFoundError:
                print("FileNotFoundError: {}/{}".format(self.root, self.list[index]))
                return self[index+1]

        # >>> Check that we're training with that object
        # obj = meta['cls_indexes'].flatten().astype(np.int32)
        # if self._remapped_getitem is None:
        #     object_exists = False
        #     for obj_idx in obj:
        #         if obj_idx in self.cld:
        #             object_exists = True
        #     if not object_exists:
        #         return None
        #     return True

        try:
            if self.list[index][:8] != 'data_syn' and int(self.list[index][5:9]) >= 60:
                cam_cx = self.cam_cx_2
                cam_cy = self.cam_cy_2
                cam_fx = self.cam_fx_2
                cam_fy = self.cam_fy_2
            else:
                cam_cx = self.cam_cx_1
                cam_cy = self.cam_cy_1
                cam_fx = self.cam_fx_1
                cam_fy = self.cam_fy_1
        except ValueError:
            return self[index-1]

        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        add_front = False
        if self.add_noise:
            for k in range(5):
                seed = random.choice(self.syn)
                # temp_img = self.load('{0}/{1}-color.png'.format(self.root, seed))
                try:
                    temp_img = Image.fromarray(self.list_rgb[:, :, :, int(seed)])
                except IndexError:
                    temp_img = Image.fromarray(self.list_rgb[:, :, :, int(seed-1)])

                front = np.array(self.trancolor(temp_img).convert("RGB"))
                front = np.transpose(front, (2, 0, 1))
                # f_label = np.array(self.load('{0}/{1}-label.png'.format(self.root, seed)))
                try:
                    f_label = self.list_label[:, :, int(seed)]
                except IndexError:
                    f_label = self.list_label[:, :, int(seed-1)]

                front_label = np.unique(f_label).tolist()[1:]
                if len(front_label) < self.front_num:
                   continue
                front_label = random.sample(front_label, self.front_num)
                for f_i in front_label:
                    mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
                    if f_i == front_label[0]:
                        mask_front = mk
                    else:
                        mask_front = mask_front * mk
                t_label = label * mask_front
                if len(t_label.nonzero()[0]) > 1000:
                    label = t_label
                    add_front = True
                    break

        obj = meta['cls_indexes'].flatten().astype(np.int32)

        while 1:
            idx = np.random.randint(0, len(obj))
            # >>> Make sure that object is one of the ones we're loading
            if obj[idx] not in self.cld:
                continue
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) > self.minimum_num_pt:
                break

        if self.add_noise:
            img = self.trancolor(img)

        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

        if self.list[index][:8] == 'data_syn':
            seed = random.choice(self.real)
            # temp_img = self.load('{0}/{1}-color.png'.format(self.root, seed))
            try:
                temp_img = Image.fromarray(self.list_rgb[:, :, :, int(seed)])
            except IndexError:
                temp_img = Image.fromarray(self.list_rgb[:, :, :, int(seed-1)])
            back = np.array(self.trancolor(temp_img).convert("RGB"))
            back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
            img_masked = back * mask_back[rmin:rmax, cmin:cmax] + img
        else:
            img_masked = img

        if self.add_noise and add_front:
            img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + front[:, rmin:rmax, cmin:cmax] * ~(mask_front[rmin:rmax, cmin:cmax])

        if self.list[index][:8] == 'data_syn':
            img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

        # p_img = np.transpose(img_masked, (1, 2, 0))
        # scipy.misc.imsave('temp/{0}_input.png'.format(index), p_img)
        # scipy.misc.imsave('temp/{0}_label.png'.format(index), mask[rmin:rmax, cmin:cmax].astype(np.int32))

        target_r = meta['poses'][:, :, idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')
        
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam_scale = meta['factor_depth'][0][0]
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        if self.add_noise:
            cloud = np.add(cloud, add_t)

        # fw = open('temp/{0}_cld.xyz'.format(index), 'w')
        # for it in cloud:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()

        dellist = [j for j in range(0, len(self.cld[obj[idx]]))]
        if self.refine:
            dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_large)
        else:
            dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_small)
        model_points = np.delete(self.cld[obj[idx]], dellist, axis=0)

        # fw = open('temp/{0}_model_points.xyz'.format(index), 'w')
        # for it in model_points:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()

        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t + add_t)
        else:
            target = np.add(target, target_t)
        
        # fw = open('temp/{0}_tar.xyz'.format(index), 'w')
        # for it in target:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()
        
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([int(obj[idx]) - 1])

    def __len__(self):
        return self.length
        # >>> Account for skipped samples
        # if self._remapped_getitem is None:
        #     return self.length
        # return len(self._remapped_getitem)

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
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
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax
