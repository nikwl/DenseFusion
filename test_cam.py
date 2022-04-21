import argparse
import os

import numpy as np
import open3d as o3d
from PIL import Image
import cv2


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
    config = o3d.io.AzureKinectSensorConfig()
    sensor = o3d.io.AzureKinectSensor(config)
    if not sensor.connect(0):
        raise RuntimeError('Failed to connect to sensor')
    
    idx = 1
    while True:
        # Capture a frame
        rgbd = sensor.capture_frame(True)

        if rgbd is None:
            continue

        color = np.array(rgbd.color)
        depth = np.array(rgbd.depth, dtype=np.uint16)

        color = Image.fromarray(color).resize((640, 480))
        depth = Image.fromarray(depth).resize((640, 480))

        if args.visualize:
            cv2.imshow(
                "kinect stream", 
                cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            )

        if args.output is not None:
            color.save(
                os.path.join(args.output, "color_" + str(idx) + ".png")
            )
            depth.save(
                os.path.join(args.output, "depth_" + str(idx) + ".png")
            )
        idx += 1
