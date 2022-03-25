# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/sensors/azure_kinect_viewer.py

import numpy as np
import open3d as o3d
from PIL import Image
import cv2


config = o3d.io.AzureKinectSensorConfig()
sensor = o3d.io.AzureKinectSensor(config)
if not sensor.connect(0):
    raise RuntimeError('Failed to connect to sensor')


while True:
    rgbd = sensor.capture_frame(True)

    if rgbd is None:
        continue

    color = np.array(rgbd.color)
    depth = np.array(rgbd.depth)

    color = np.array(Image.fromarray(color).resize((640, 480)))
    depth = np.array(Image.fromarray(depth).resize((640, 480)))

    cv2.imshow("test", cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(1) == 27: 
        exit()
