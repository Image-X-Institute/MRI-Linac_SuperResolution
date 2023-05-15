# MRI-Linac_SuperResolution
## Integrating deep learning into an MRI-linac

Authors: **James Grover**

Contained here is the code and documentation required to integrate super-resolution into Gadgetron. 

## Build
### Prerequisites:
#### Gadgetron, Docker, CUDA.
[Gadgetron](https://github.com/gadgetron/gadgetron) is distributed using Docker containers. A working Docker installation is required. In addition, the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is required to allow Gadgetron (and the deep learning interface) to utilise CUDA GPUs. 

#### PyTorch (and other Python dependencies)
This is done in the *Dockerfile_base_ai_image*. Tweaking of the PyTorch / CUDA toolkit version may be needed when using different hardware.

#### Obtaining trained parameters
In this repository under releases are the trained model parameters. These will need to be downloaded and placed under code/modules/parameters/.
* 64_to_256_bicubic_interpolation_JIT.pt    --    the bicubic interpolation model.
* 2022-09-10_11-22-39_edsr_nonoise.pt    --    the EDSR trained on brains.

## Building
Once Docker is installed with the NVIDIA container runtime we can build a Gadgetron (with deep learning interface) image.
First, build a general Gadgetron AI base image. This image simply has the Python dependencies installed.
```sh
docker build -f Dockerfile_base_ai_image -t gadgetron_ai_base:0.1 .
```

Second, build the super-resolution image (based on the Gadgetron base ai image built above).
```sh
docker build -t gadgetron_ai:0.1 .
```

## Running
**The official Gadgetron GitHub repository contains a readthedocs that can be useful.** 
A basic example run command is provided below.
When deploying online, port mapping will need to enabled.
When saving reconstructions for offline analysis (e.g., on the host machine), volume mounting will need to be enabled. 

```sh
docker run --gpus=all -ti --name=gadgetron --rm gadgetron_ai:0.1
```

## Testing
Once Gadgetron is running, open a new terminal, **leave the terminal where you did docker run ... running**. 
```sh
docker exec -ti gadgetron bash
```


In this connected shell, navigate to /tmp (a good place to store reconstructions). We can generate some test data using an inbuilt Shepp-Logan phantom generator.
```sh
ismrmrd_generate_cartesian_shepp_logan -r 10 -m 64
```
This will create a testdata.h5 file, simulating an acquisition. For super-resolution methods in this repository it's important the matrix size is 64x64 so don't forget the -m 64 argument!


Test a standard reconstruction with **no** deep learning.
```sh
gadgetron_ismrmrd_client -f testdata.h5 -c GrappaTrackingDisabled.xml
```


Test a deep learning-based reconstruction.
```sh
gadgetron_ismrmrd_client -f testdata.h5 -c GrappaEdsrTrackingDisabled.xml
```


Can also test the integration into MLC tracking using the test server script (locating in test/). 
Run this script locally on the host machine (note: matplotlib and numpy will need to be installed):
```python
python test_mlc_tracking_server.py -s 64 -p
```

We also need to modify our run command (specify the additional network=host switch).

```sh
docker run --network=host --gpus=all -ti --name=gadgetron --rm gadgetron_ai:0.1
```

From here, we can generate the Shepp-Logan phantom as before and run reconstructions with the tracking enabled:
```sh
gadgetron_ismrmrd_client -f testdata.h5 -c GrappaTrackingEnabled.xml
```

With super-resolution:
On the host machine we need to re-run the server script with updated image dimensions (i.e. 256):
```python
python test_mlc_tracking_server.py -s 256 -p
```
In the Docker container:
```sh
gadgetron_ismrmrd_client -f testdata.h5 -c GrappaEdsrTrackingEnabled.xml
```

Meta-data should stream to the test_mlc_tracking_server script and a plot should be generated of the first acquisition (if the -p switch was provided).

## Directory Structure
* code/ - Deep learning framework source code. 
* test/ - Contains a simple test multi-leaf collimator (MLC) tracking server script.
* .dockerignore
* Dockerfile_base_ai_image - This is used to build the Docker image with DL dependencies.
* Dockerfile - This is used to build the Docker image with the deep learning framework.
* LICENSE
* README.md

## Acknowledgement
EDSR was trained on brain data from the QIN-GBM Treatment Response Cancer Imaging Archive dataset.
1. Mamonov A, Kalpathy-Cramer J. Data From QIN GBM Treatment Response. The Cancer Imaging Archive. 2016;
2. Prah M, Stufflebeam S, Paulson E, et al. Repeatability of standardized and normalized relative CBV in 
patients with newly diagnosed glioblastoma. American Journal of Neuroradiology. 2015;36(9):1654-1661.
3. Clark K, Vendt B, Smith K, et al. The Cancer Imaging Archive (TCIA): maintaining and operating a public 
information repository. Journal of digital imaging. 2013;26(6):1045-1057.
