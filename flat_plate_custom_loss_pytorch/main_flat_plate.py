import torch.cuda
from sympy import Symbol, cos, sin
import modulus
import numpy as np
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.solver import Solver
from modulus.models.fully_connected import FullyConnectedArch
# from modulus.domain.inferencer import PointwiseInferencer
from modulus.key import Key
# from modulus.utils.io import InferencerPlotter
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:32240'

# misc files imports
from geometry_create import create_geometry
from poison_2d import Poison_2D
from domain_define import define_domain


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # ----------- Poisson 2D ----------- #
    poisson_2d = Poison_2D(cfg)
    # flow_net = instantiate_arch(
    #     input_keys=[Key("x"), Key("y"), Key("alpha")],
    #     output_keys=[Key("u"), Key("v"), Key("phi")],
    #     layer_size=512,
    #     nr_layers=6,
    #     cfg=cfg.arch.fully_connected  # six hidden layers with 512 neurons per layer.
    # )
    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("alpha")],
        output_keys=[Key("u"), Key("v"), Key("phi")],
        layer_size=50,
        nr_layers=6,
        # cfg=cfg.arch.fully_connected  # six hidden layers with 512 neurons per layer.
    )
    arch = flow_net.make_node(name="flow_network")
    nodes = poisson_2d.make_nodes() + [arch]
    # ----------- Poisson 2D ----------- #

    # domain height and weight
    height = cfg.custom.unscaled_domain_height * cfg.custom.obstacle_length
    width = cfg.custom.unscaled_domain_width * cfg.custom.obstacle_length

    # create symbolic variables
    x, y, alpha = Symbol('x'), Symbol('y'), Symbol('alpha')

    # Define range for AoA (alpha) range: (for instance its -10 to 10 degrees)
    alpha_range = {
        alpha: lambda batch_size: np.full((batch_size, 1),
                                          np.random.uniform(- np.pi * cfg.custom.free_stream_velocity / 180,
                                                            np.pi * cfg.custom.free_stream_velocity / 180))}

    # velocity components in the flow fields.
    u_x = cfg.custom.free_stream_velocity * cos(alpha)  # 10 * cos(alpha)
    u_y = cfg.custom.free_stream_velocity * sin(alpha)  # 10 * sin(alpha)

    # make geometry
    geo, obstacle_above, obstacle_below, wake1_above, wake1_below, wake2_above, \
    wake2_below, wake3_above, wake3_below = create_geometry(cfg)

    # make ldc domain
    flat_plate_domain = define_domain(cfg, geo, obstacle_above, obstacle_below, wake1_above, wake1_below, wake2_above,
                                      wake2_below, wake3_above, wake3_below, alpha_range, nodes, height, width, u_x,
                                      u_y, arch)

    # add inference data

    # ----- Inference ----- #
    # mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
    # openfoam_var = csv_to_dict(
    #     to_absolute_path("openfoam/cylinder_nu_0.020.csv"), mapping
    # )
    # openfoam_invar_numpy = {
    #     key: value for key, value in openfoam_var.items() if key in ["x", "y"]
    # }
    # grid_inference = PointwiseInferencer(
    #     nodes=nodes,
    #     invar=openfoam_invar_numpy,
    #     output_names=["u", "v", "p"],
    #     batch_size=1024,
    #     plotter=InferencerPlotter(),
    # )
    # ldc_domain.add_inferencer(grid_inference, "inf_data")
    # ----- Inference ----- #

    # ----- Validator ----- #
    # openfoam_validator = PointwiseValidator(
    #     nodes=nodes, invar=openfoam_invar_numpy, true_outvar=openfoam_outvar_numpy
    # )
    # domain.add_validator(openfoam_validator)
    # ----- Validator ----- #
    # make solver
    slv = Solver(cfg, flat_plate_domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
