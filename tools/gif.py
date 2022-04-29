import argparse
import os
import math
import numpy as np
from PIL import Image
import trimesh
import json
import utils_3d
import cv2
import tqdm

CLASSES_FILE = 'datasets/ycb/dataset_config/classes.txt'
OBJECTS_DIR = 'models'


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < 1e-5:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


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


def depth_mask(depth, thresh=500):
    return cv2.morphologyEx(
        (np.logical_and(depth < thresh, depth > 5)).astype(float), 
        cv2.MORPH_OPEN, 
        np.ones((3,3)),
    )


def vis_pose(
    object_model,
    color,
    rotation, 
    translation,
):

    rotation = np.array(rotation)
    translation = np.array(translation)

    my_r = quaternion_matrix(rotation)[:3, :3]
    my_t = translation

    # rotation = quaternion_matrix(rotation)
    # rotation[:3, :3] = rotation[3, :3].T
    # translation = trimesh.transformations.translation_matrix(translation)
    # object_model.apply_transform(
    #     translation @ rotation
    # )

    object_model.vertices = np.dot(object_model.vertices, my_r.T) + my_t
    object_model.fix_normals()

    # 90, 180, 270

    # for x in [0, 90, 180, 270]:
    #     for y in [0, 90, 180, 270]:
    #         for z in [0, 90, 180, 270]:

    # Render
    mesh_render = utils_3d.render_mesh(
        object_model,
        bg_color=0,
        resolution=color.shape[:2][::-1],
        xrot=180,
        yrot=0,
        zrot=0,
    )
    # output_img = np.hstack((
    #     color,
    #     mesh_render,
    # ))

    # Image.fromarray(output_img).save("{}_{}_{}.png".format(x, y, z))
    return mesh_render


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='input images')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--object_idx', type=int, default = 15, help='object index')
parser.add_argument('--start_at', type=int, default = 0, help='')
parser.add_argument('--end_after', type=int, default = 1000, help='')
parser.add_argument('--ycb_format', action='store_true', default=False)
parser.add_argument('--linemod_format', action='store_true', default=False)
opt = parser.parse_args()

dataset_config_dir = 'datasets/ycb/dataset_config'

assert os.path.isdir(opt.input)


if opt.ycb_format:
    object_model = load_object(opt.object_idx)
    object_model = utils_3d.normalize_unit_cube(object_model)
    
    color_template = "{}-color.png"
    depth_template = "{}-depth.png"
    tf_template = "{}-tf.json"

    testlist = []
    input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
        testlist.append(input_line)
    input_file.close()
elif opt.linemod_format:
    fnames = [p.split(".")[0] for p in os.listdir(os.path.join(opt.input, "rgb"))]
    fnames = sorted(fnames)

    color_template = "rgb/{}.png"
    depth_template = "depth/{}.png"
    tf_template = "rgb/{}.json"

    path = "datasets/linemod/Linemod_preprocessed/models/obj_01.ply"
    print("Loading model from: {}".format(
        path
    ))
    object_model = trimesh.load(path)
    object_model = utils_3d.normalize_unit_cube(object_model)

else:
    object_model = load_object(opt.object_idx)
    object_model = utils_3d.normalize_unit_cube(object_model)
    
    color_template = "color_{}.png"
    depth_template = "depth_{}.png"
    tf_template = "tf_{}.json"

counter = opt.start_at
img_list = []
pbar = tqdm.tqdm()
while 1:
    if opt.ycb_format:
        pointer = testlist[counter+1]
    elif opt.linemod_format:
        pointer = fnames[counter]
    else:
        pointer = counter

    counter += 1
    pbar.update(1)

    if counter > (opt.start_at + opt.end_after):
        break

    try:
        img = Image.open(
            os.path.join(opt.input, color_template.format(pointer))
        )
        
        depth = np.array(Image.open(
            os.path.join(opt.input, depth_template.format(pointer))
        ))
        
        tf = json.load(
            open(
                os.path.join(opt.input, tf_template.format(pointer)),
                "r",
            )
        )
    except FileNotFoundError:
        continue

    if tf is None:
        continue

    pose_img = vis_pose(
        object_model.copy(),
        np.array(img),
        **tf,
    )

    img_list.append(
        np.hstack((
            np.array(img),
            pose_img,
            # ((np.array(img).astype(float) + pose_img) / 2).astype(np.uint8),
            # utils_3d.colorize_image(depth_mask(depth), cmap="gray")[:, :, :3]
        )), 
    )

utils_3d.save_gif(os.path.basename(opt.output), img_list)
