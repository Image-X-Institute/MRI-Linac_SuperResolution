# MRI-Linac_SuperResolution
## Integrating deep learning into an MRI-linac

Authors: **James Grover**

Contained here is the code and documentation required to integrate AI into Gadgetron. 
These will need to be downloaded (**under releases**) and saved under in code/modules/parameters/

## Setup / Build / Install
### Prerequisites:
#### Gadgetron, Docker, CUDA.
[Gadgetron](https://github.com/gadgetron/gadgetron) is distributed using Docker containers. A working Docker installation is required. In addition, the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is required to allow Gadgetron (and the deep learning interface) to utilise CUDA GPUs. 

#### PyTorch
Conda install to install additional dependencies into the Gadgetron Docker image. This means each time we run the Docker image, we have PyTorch (and additional dependencies) working out of the box. This is done in the *Dockerfile*. Tweaking of the PyTorch / CUDA toolkit version may be needed when using different hardware.

#### Building
Once Docker is installed with the NVIDIA container runtime we can build a Gadgetron (with deep learning interface) image.
First, build a general Gadgetron AI base image. This image simply has the Python dependencies installed.
```sh
docker build -f Dockerfile_base_ai_image -t gadgetron_ai_base:0.1 .
```

Second, build the super-resolution image.
```sh
docker build -t gadgetron_ai:0.1 .
```

## Running
**The official Gadgetron GitHub repository contains a readthedocs that can be useful.** 
An example run command is provided below, note modification will be required for your specific system (especially the --volume argument). 
When deploying online with the GadgetronICE, port mapping will need to be enabled.

```sh
docker run --gpus=all -ti --name=gadgetron --volume=/home/james/gadgetron_data:/tmp/gadgetron_data --rm gadgetron_ai:0.1
```
To see running containers, run:
```sh
docker ps
```
To stop a running container, run: **WARNING: once a container is stopped (and removed), data (i.e., reconstructions) does not generally persist! Make sure copies of reconstructions to keep have been saved through the volume mount before stopping a container.**
```sh
# If you named the container gadgetron in the docker run ... command, otherwise use the container ID from docker ps.
docker stop gadgetron
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
I recommend testing that the transfer went smoothly by testing a simple reconstruction that **does not** invoke the deep learning framework.
```sh
gadgetron_ismrmrd_client -f testdata.h5 -c GrappaTrackingDisabled.xml
```
If this works, then I recommend testing a reconstruction that **does** invoke the deep learning framework.
```sh
gadgetron_ismrmrd_client -f testdata.h5 -c GrappaEdsrTrackingDisabled.xml
```
You will know if all went smoothly by checking the output of the terminal where the docker run ... command was invoked (where the Gadgetron output lies).
I also recommend moving the out.h5 through the mounted volume for analysis with a Jupyter notebook or similar on the host machine.

Can also test the integration into MLC tracking using the test server script (locating in test/). 
Run this script locally on the host machine:
```python
python test_mlc_tracking_server.py -s 64 -p
```
**Reminder the -s argument specifies the size of the incoming image, change accordingly (i.e., -s 256 when using super-resolution).**

We also need to modify our run command (specify the additional network=host switch).

```sh
docker run --network=host --gpus=all -ti --name=gadgetron --volume=/home/james/gadgetron_data:/tmp/gadgetron_data --rm gadgetron_ai:0.1
```

From here, we can generate the Shepp-Logan phantom as before and run reconstructions with the tracking enabled:
```sh
gadgetron_ismrmrd_client -f testdata.h5 -c GrappaTrackingEnabled.xml
```

With super-resolution:
```sh
gadgetron_ismrmrd_client -f testdata.h5 -c GrappaEdsrTrackingEnabled.xml
```

Meta-data should stream to the test_mlc_tracking_server script and an plot should be generated of the first acquisition (if the -p switch was provided).

## Directory Structure
* code/ - This is where the deep learning framework lives. These files are copied to their appropriate location in the Dockerfile.
* .dockerignore
* Dockerfile - This is used to build the Docker image.
* LICENSE
* README.md
