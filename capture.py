import argparse
import os

import numpy as np
import open3d as o3d
from matplotlib import cm
from PIL import Image
import cv2


def colorize_image(img, cmap="jet", vmin=None, vmax=None):
    assert len(img.shape) < 3

    # Get the min and max
    if vmin is None:
        vmin = img.min()
    if vmax is None:
        vmax = img.max()

    # Clip and rescale
    img = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)

    if cmap is None or cmap == "None":
        return (np.dstack((img, img, img, np.ones((img.shape)))) * 255).astype(np.uint8)

    cmap = cm.get_cmap(cmap)

    # Apply the colormap
    return (cmap(img) * 255).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interface with the kinect and save a sequence of images "
        + "to a directory."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Location to save the output images to.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the image stream.",
    )
    args = parser.parse_args()
    if args.output is not None:
        assert os.path.isdir(args.output)

    # Connect to kinect
    # config = o3d.io.AzureKinectSensorConfig()
    config = o3d.io.read_azure_kinect_sensor_config("config.json")
    sensor = o3d.io.AzureKinectSensor(config)
    if not sensor.connect(0):
        raise RuntimeError('Failed to connect to sensor')
    
    idx = 1
    while True:
        # Capture a frame
        rgbd = sensor.capture_frame(True)

        if rgbd is None:
            continue
        
        # cd = rgbd.create_from_color_and_depth(rgbd.color, rgbd.depth)
        # print(dir(cd))
        # print(np.array(cd.color).shape)
        # print(np.array(cd.depth).shape)
        # exit()

        # color = cv2.resize(
        #     np.array(rgbd.color), 
        #     (640, 480)
        # )
        color = np.array(rgbd.color)
        depth = np.array(rgbd.depth, dtype=np.uint16)
        # print(depth.shape)
        # print(color.shape)

        # print(dir(rgbd.PointCloud))
        # print(rgbd.PointCloud.value)

        # rgbd.streamCloud()
        # exit(0)

        if args.visualize:
            # d = depth
            d = cv2.resize(
                depth, 
                color.shape[:2][::-1]
            )
            d = cv2.morphologyEx(
                (np.logical_and(d < 500, d > 10)).astype(float), 
                cv2.MORPH_OPEN, 
                np.ones((3,3)),
            )
            cv2.imshow(
                "kinect stream", 
                np.hstack((
                    cv2.cvtColor(color, cv2.COLOR_BGR2RGB),
                    colorize_image(d, cmap="gray")[:, :, :3],
                ))
            )
            if cv2.waitKey(33) == ord("q"):
                break

        if args.output is not None:
            Image.fromarray(color).save(
                os.path.join(args.output, "color_" + str(idx) + ".png")
            )
            Image.fromarray(depth).save(
                os.path.join(args.output, "depth_" + str(idx) + ".png")
            )
        idx += 1
