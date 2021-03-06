# Pose-Estimation

![](assets/drill.gif)
![](assets/ycb_3.gif)
![](assets/ycb_10.gif)
<!-- ![](assets/ycb_15.gif) -->

Final report is in report.pdf

## Installation

Tested on Ubutnu 18.04 with python 3.6.9. We also include instructions on how to install wsl in case you need to train or test the approach and do not have access to an ubuntu machine.
NOTE: Azure kinect sensor is required.

Use [this mirror](https://okabe.dev/ycb-video-dataset-download-mirror/) to download the YCB_Video dataset.

### (1) (Optional) Using Windows via WSL

PCs need to be updated to at minimum <b>Version 21H2 (OS Build 19041)</b> \
To check the version, open the Run window (win+r), and run `winver`. \
Use windows update to update to the correct version. You may need to check for updates and restart multiple times. \

Open a powershell instance with administrator privileges. Then run:
```powershell
# Install wsl
wsl --install -d Ubuntu
```

Create a default user inside of wsl2. Run the following in powershell to start wsl:
```powershell
wsl
```

NOTE: If `sudo apt-get update` fails to resolve links, change the nameserver version in /etc/resolv.conf to 8.8.8.8:
```powershell
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf > /dev/null
```

Install docker desktop from [HERE](https://www.docker.com/products/docker-desktop/). Then go to Settings > Resources > WSL INTEGRATION and enable docker inside of the `Ubuntu` distro. See [this guide for troubleshooting](datasets/linemod/Linemod_preprocessed/data/01/depth/0001.pnghttps://docs.docker.com/desktop/windows/wsl/).

### (2) Installing the Azure Kinect

Install apt dependencies (OpenSSL and OpenGL).
```bash 
sudo apt-get install libssl-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev
```

Configure Microsoft's Software repository. \
Follow the [official installation instructions](https://docs.microsoft.com/en-us/azure/kinect-dk/sensor-sdk-download)
```bash
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
sudo apt-get update
```

Install the Azure Kinect Firmware Tool.
```bash
sudo apt install k4a-tools
sudo apt install libk4a1.4-dev
```

Sometimes even after the default installation the device will fail to initialize. If this happens, perform the following step.
```bash
cp scripts/99-k4a.rules /etc/udev/rules.d/
```

Create a python environment for using the kinect.
```bash
virtualenv -p python3.6 env
```

Install the python dependencies.
```bash
pip install open3d opencv-python pillow
```

Now test the camera.
```bash
python capture.py --visualize
```

### (3) Installing nvidia-docker

Make sure curl is installed.
```bash
sudo apt install curl
```

Install docker using the [convenience script](https://docs.docker.com/engine/install/ubuntu/). DO NOT do this step if you've installed docker desktop.
```bash 
curl https://get.docker.com | sh \
    && sudo systemctl --now enable docker
```

Install nvidia-docker following the [official installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). \
Setup the package repository and the GPG key.
```bash 
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Install the nvidia-docker package.
```bash
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

Restart docker.
```bash
sudo systemctl restart docker
```

Test the installation.
```bash
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

This should result in a console output shown below:
```bash
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   34C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### (3) Installing our docker image

Pull our docker image
```bash
sudo docker pull nikwl/densefusion
```

Mount the local directory and create an interactive docker container.
```bash
#!/bin/bash
REFNAME="densefusion"
REPONAME="nikwl/$REFNAME"
MY_USERNAME=`whoami`

docker run \
    -it \
    --gpus all \
    --network host \
    --volume `pwd`:/opt/ws \
    --volume /home/nikolas/dev/data4:/opt/data \
    --name "$REFNAME" \
    $REPONAME \
    bash
```

Once you've entered into the docker container, run the setup script to pull the most recent commit.
```bash
./setup.sh
```

Then run the following commands (TODO: roll into the docker image)
```
cd opt/ws/DenseFusion
pip install gdown trimesh pillow==6.1 pyrender
python download.py
./download.sh
apt install -y freeglut3-dev freeglut3 libgl1-mesa-dev libglu1-mesa-dev libxext-dev libxt-dev
apt install -y python3-opengl libgl1-mesa-glx libglu1-mesa
```

## (4) Segmentation 

Segmentation ended up being a hosted in a [different repo](https://github.com/desichic/Segmentation). Please follow its respective installation instructions.

## Operating procedure

Create a test directory to store the images in.
```bash
mkdir trial_run
```

Record some images.
```bash
python capture.py --output trial_run --visualize
```

Run the segmentation algorithm. 
```bash
-
```

Run the pose estimation algorithm in docker. By default it will use depth masking.
```bash
./eval_real.sh trial_run
```

Run the pose visualization.
```bash
./gif_real.sh trial_run
```

Open up the `trial_run.gif` to see the results.

## Results
Left: input RGB. Middle: depth mask. Right: oriented model.
![](assets/test_capture2.gif)
