# Solving aspects of aerodynamics using NVIDIA Modulus

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [License](https://github.com/neo-fetch/nvidia-modulus-airfoil-optimisation/blob/master/LICENSE)
- [Acknowledgements](#acknowledgements)

## About <a name = "about"></a>

This repository uses modulus deeplearning framework to solve some of the different aspects of aerodynamics seen in a flat plate scenario for different speeds, dealing from laminar flow conditions all the way to subsonic and supersonic pockets. We however, use Physics Informed Neural Networks (PINNs) to solve the aerodynamics of the aircraft. The PINNs are implemented in the modulus framework and can be found [here](https://docs.nvidia.com/deeplearning/modulus/)

Instead of the traditional Navier stokes provided by the modulus libraries, we use our own implementation of navier stokes(More on this [here](https://ieeexplore.ieee.org/document/9003058). Please refer to section `III. NUMERICAL MODEL OF FLUID DYNAMICS` of the paper for more details on the problem setup.).
```
     +---------+
    /|/     \|/|
   //|// ---...|
  ///|/////////|
 ////+---------+
 //////////////
 / ////////////
   / //////////
where '/', '|' and '\' are u + v such that tan-1(v/u) = x degrees(ranges from  -10 to +10)
```

On using the modulus framework for our problem setup, we encounter different problems such as:

- Enforcing the [kutta condition](https://en.wikipedia.org/wiki/Kutta_condition) by constructing a wakeline that allows gradual release of velocity in the y direction.

- Training a neural network solver for a genralized range of angle of attack( here the angle of attack ranges from -10 to +10 degrees) by finding methods to generalize the angle as a parameter.

- Dealing with discontinuities inside the setup by using variational methods, custom loss, KD-Trees and filtered differentiation.

Functional values may need to be differentiated at points close to discontinuous boundaries, where information cannot be allowed to flow across discontinuities. This means that derivatives at points on the other side of a discontinuous surface cannot be obtained by using points on the opposite side of the surface as a reference. 

The process of getting derivatives is called "filtered differentiation." This means that at certain points in the process, the values at certain points are not taken into account when getting derivatives. Due to its automatic differentiation process, Modulus software can't do this because it doesn't have any mechanisms known for filtering.

As a result, we developed a novel approach that enables the implementation of Filtered Differentiation (FD) at specific points. Obviously, this results in longer computation times; thus, one of the principles followed in this implementation is to perform FD evaluation only at necessary points.


## Getting Started <a name = "getting_started"></a>

Please visit [NVIDIA Modulus](https://developer.nvidia.com/modulus) for more information.

### Prerequisites

- [NVIDIA Modulus (Previously known as NVIDIA SimNet)](https://developer.nvidia.com/modulus)
- [Docker](https://www.docker.com/)
- [Python](https://www.python.org/) with [sympy](https://www.sympy.org/), [numpy](https://www.numpy.org/) and [matplotlib](https://matplotlib.org/)
- [Paraview](https://www.paraview.org/)
- Physics and Math (Fluid dynamics, Computational Fluid Dynamics, Numerical methods and basic Machine Learning and training Neural Networks)

### Installing

A step by step installation guide can be found [here](#getting_started), however this guide feels outdated, as it recommends using nvidia-docker. 

Because I use Arch Linux, Below are the steps I followed to run the docker image. Although I suspect that the overall steps dont change from distro to distro and remain mostly the same(I have not tested this but I'm confident as docker is infrastructure agnostic).

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
git clone https://www.github.com/neo-fetch/nvidia-modulus-airfoil-optimisation.git; cd nvidia-modulus-airfoil-optimisation
```
- Run the python script

```
python src/ldc_2D.py --layer_size 100 --nr_layers 2
```

## Results

The results can be plotted using paraview or numpy and matplotlib.

The convergence contour plots can be found in the `results` folder.

## Acknowledgements <a name = "acknowledgements"></a>

- [Mayank Deshpande](https://www.github.com/neo-fetch)
- [Siddharth Agarwal](https://www.linkedin.com/in/siddharthagarwal1089/)
- [Yang Juntao](https://sg.linkedin.com/in/yang-juntao-b0734359)
- [Neelappa H](https://github.com/neelu065)
