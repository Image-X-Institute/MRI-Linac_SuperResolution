# MRI-Linac_SuperResolution
## Integrating deep learning into an MRI-linac

Authors: **James Grover**

Contained here is the code and documentation required to integrate super-resolution into Gadgetron. 

## Setup / Build / Install
### Prerequisites:
#### Gadgetron, Docker, CUDA.
[Gadgetron](https://github.com/gadgetron/gadgetron) is distributed using Docker containers. A working Docker installation is required. In addition, the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is required to allow Gadgetron (and the deep learning interface) to utilise CUDA GPUs. 

#### PyTorch
Conda install to install additional dependencies into the Gadgetron Docker image. This means each time we run the Docker image, we have PyTorch (and additional dependencies) working out of the box. This is done in the *Dockerfile*. Tweaking of the PyTorch / CUDA toolkit version may be needed when using different hardware.

#### Obtaining trained parameters
In this repository under releases are the trained model parameters. These will need to be downloaded and placed under code/modules/parameters/.
* 64_to_256_bicubic_interpolation_JIT.pt    --    the bicubic interpolation model.
* 2022-09-10_11-22-39_edsr_nonoise.pt    --    the EDSR trained on brains.
* 2022-09-15_08-58-43_edsr_thorax_nonoise.pt    --    the EDSR fine-tuned to thoraxes.

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
An example run command is provided below, note modification will be required for your specific system (especially the --volume argument). 
When deploying online with the GadgetronICE, port mapping will need to be enabled.

```sh
docker run --gpus=all -ti --name=gadgetron --rm gadgetron_ai:0.1
```

## Testing
Now Gadgetron is ready and waiting. To test our interface we need to connect a shell to this container. Open a new terminal, **leave the terminal where you did docker run ... running**. 
```sh
docker exec -ti gadgetron bash
```
In this connected shell, navigate to (i.e., cd /tmp). We can generate some test data using an inbuilt Shepp-Logan phantom generator.
```sh
ismrmrd_generate_cartesian_shepp_logan -r 10 -m 64
```
This will create a testdata.h5 file, simulating an acquisition.


Test a standard reconstruction with **no** deep learning.
```sh
gadgetron_ismrmrd_client -f testdata.h5 -c GrappaTrackingDisabled.xml
```


Test a deep learning-based reconstruction.
```sh
gadgetron_ismrmrd_client -f testdata.h5 -c GrappaEdsrTrackingDisabled.xml
```

The out.h5 file can be inspected on the host machine using volume mounts (this would mean modifying the run command).

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
