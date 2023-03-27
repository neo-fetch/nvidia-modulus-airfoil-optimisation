> File execution steps:

1) Log in to the HPC and access the modulus 22.07 container using the following command:  
`docker run --shm-size=50g --ulimit memlock=-1 --ulimit stack=67108864 --cpuset-cpus=0-39 -it --rm -v $HOME:$HOME -e DISPLAY=10.59.115.20:0.0 -e PYCHARM_PYTHON_PATH=/opt/conda/bin/python3 -e PATH=$PATH:/home/u10140/pycharm-community-2022.2.1/bin/ nvcr.io/nvidia/modulus/modulus:22.07`  
If any problems exists, please check the existence of modulus container and/or reisntall modulus 22.07 container.

2) Open pycharm within this folder, and run it.

> Issues yet to resolve:

- Although, the boundary conditions and even the physics are implemented correctly, there is a inherent issue with modulus which need to be resolved. For example, for problem like this, there is a requirement for 'alpha' to be same at all co-ordinate points for given epoch. 
- However, after multiple attempts we could not set the `same alpha for all points in a domain` and `alpha need to vary in every epoch`. Training in this way, for multiple epochs should result in accurate prediction of flow over a flat plate. 

> Further details on this issue contact: [neelu065[at].gmial.com](http://neelu065[at].gmial.com "neelu065[at].gmial.com")
