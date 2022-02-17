# Flat Plate simulation using NVIDIA Modulus

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [License](https://github.com/neo-fetch/modulus_stuff/blob/master/LICENSE)
- [Acknowledgements](#acknowledgements)

## About <a name = "about"></a>

This project uses the flat plate scenario where a flow incidents the flat plate at an angle. Instead of the traditional Navier stokes provided by the modulus libraries, we use our own implementation of navier stokes.

## Getting Started <a name = "getting_started"></a>

Please visit [NVIDIA Modulus](https://developer.nvidia.com/modulus) for more information.

### Prerequisites

- [NVIDIA Modulus](https://developer.nvidia.com/modulus)
- [Docker](https://www.docker.com/)
- [Python](https://www.python.org/)
- Physics and Math (Fluid dynamics, Computational Fluid Dynamics, and Numerical methods)

### Installing

A step by step installation guide can be found [here](#getting_started), however this guide feels outdated, as it recommends using nvidia-docker. 

Because I use Arch Linux, Below are the steps I followed to run the docker image. Although I suspect that the overall steps dont change from distro to distro and remain mostly the same:

The [nvidia-docker](https://aur.archlinux.org/packages/nvidia-docker) AUR package has been deprecated in upstream because you can now use nvidia-container-toolkit in conjunction with docker 19.03's new native GPU support to use [NVIDIA accelerated docker containers](https://wiki.archlinux.org/title/Docker#Run_GPU_accelerated_Docker_containers_with_NVIDIA_GPUs) without needing nvidia-docker.

Untar the modulus examples and install the modulus docker image.

You should know that docker has an issue with the cgroup hierarchy. This is a problem with the docker daemon and not the modulus docker image. To resolve this issue, you will have to change your kernel parameters to disable the hierarchy using the following parameter:
```
systemd.unified_cgroup_hierarchy=false
```
You can find more information about this issue [here](https://bbs.archlinux.org/viewtopic.php?id=266915).

To change the kernel parameters or add more, refer to the [kernel documentation](https://www.kernel.org/doc/Documentation/sysctl/kernel.txt). 

For Arch Linux, you can refer to the [arch-wiki](https://wiki.archlinux.org/title/Kernel_parameters) for more information.

Once the following steps have been completed, you can run the docker image using the following command where you optionally extracted the examples.tar.gz file:

```
docker run --shm-size=1g -p 8888:8888 --ulimit memlock=-1 --ulimit stack=67108864 \
--runtime nvidia -v ${PWD}/examples:/examples \
-it modulus:21.06 bash
```

## Usage <a name = "usage"></a>

- Clone the repository
```
git clone https://www.github.com/neo-fetch/modulus_stuff.git; cd modulus_stuff
```
- Run the python script

```
python src/ldc_2D.py
```

### Running multiple angles

I have made a script that allows you to run my script for multiple angles. The script is located in `src/`

You can simply run it using the following command:

```
bash runmod.sh
```
You can also change the angles you want to run by altering the `src/angles.py` file and streaming the output to `src/angles` textfile.

```
python src/angles.py > src/angles
```

## Results

The following are the results of the simulation.

There are primarily two outputs we are concerned with is the velocity components of the results. The X and Y components of the velocity are shown below.

![x_component](https://i.imgur.com/5Hb9Tg8.png)

Figure 1: X Component of the velocity

![y_component](https://i.imgur.com/zTqp3FZ.png)

Figure 2: Y Component of the velocity
## Acknowledgements <a name = "acknowledgements"></a>

- [Mayank Deshpande](https://www.github.com/neo-fetch)
- [Yang Juntao](https://sg.linkedin.com/in/yang-juntao-b0734359)