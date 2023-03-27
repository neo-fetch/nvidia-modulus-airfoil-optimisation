import torch.cuda
from modulus.domain.inferencer import PointwiseInferencer
from modulus.domain.validator import PointwiseValidator
from modulus.models.multiscale_fourier_net import MultiscaleFourierNetArch
from modulus.utils.io import InferencerPlotter, ValidatorPlotter, csv_to_dict
from sympy import Symbol, cos, sin
import modulus
import numpy as np
from modulus.hydra import instantiate_arch, ModulusConfig, to_absolute_path
from modulus.solver import Solver
from modulus.models.fully_connected import FullyConnectedArch
from modulus.models.fourier_net import FourierNetArch
# from modulus.domain.inferencer import PointwiseInferencer
from modulus.key import Key
from plot_inf import PlotInference
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:32510'

# misc files imports
from geometry_create import create_geometry
from poison_2d import Poison_2D
from domain_define import define_domain


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # ----------- Poisson 2D ----------- #
    poisson_2d = Poison_2D(cfg)
    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("alpha")],
        output_keys=[Key("u"), Key("v"), Key("phi")],
        # layer_size=512,
        # nr_layers=4,
        # cfg=cfg.arch.fully_connected  # six hidden layers with 512 neurons per layer.
    )
    # flow_net = FourierNetArch(
    #     input_keys=[Key("x"), Key("y"), Key("alpha")],
    #     output_keys=[Key("u"), Key("v"), Key("phi")],
    #     frequencies=("full", [1, 5, 10]),
    #     frequencies_params=("full", [1, 5, 10])
    # )
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
                                          np.random.uniform(- np.pi * cfg.custom.AoA / 180,
                                                            np.pi * cfg.custom.AoA / 180))}

    # velocity components in the flow fields.
    u_x = cfg.custom.free_stream_velocity * cos(alpha)
    u_y = cfg.custom.free_stream_velocity * sin(alpha)

    # make geometry
    geo, obstacle, wk_above, wk_below, wall = create_geometry(cfg)

    # make ldc domain
    flat_plate_domain = define_domain(cfg, geo, obstacle, wk_above, wk_below, wall, alpha_range, nodes, height, width, u_x,
                                      u_y, arch)

    # ----- Validator ----- #
    # add validator
    # mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
    # openfoam_var = csv_to_dict(
    #    to_absolute_path("../ldc/openfoam/cavity_uniformVel0.csv"), mapping
    # )
    # openfoam_var["x"] += -0.3 / 2  # center OpenFoam data
    # openfoam_var["y"] += -0.3 / 2  # center OpenFoam data
    # openfoam_invar_numpy = {
    #    key: value for key, value in openfoam_var.items() if key in ["x", "y", "y"]
    # }
    # openfoam_outvar_numpy = {
    #    key: value for key, value in openfoam_var.items() if key in ["u", "v", "phi"]
    # }
    # openfoam_validator = PointwiseValidator(
    #    nodes=nodes,
    #    invar=openfoam_invar_numpy,
    #    true_outvar=openfoam_outvar_numpy,
    #    batch_size=1024,
    #    plotter=ValidatorPlotter(),
    # )
    # flat_plate_domain.add_validator(openfoam_validator)

    # Inference template ready ########
    # if cfg.run_mode == 'eval':
    input_points = cfg.batch_size.Inference_int_pts
    x_input = np.random.uniform(width / 2, -width / 2, size=(input_points, 1))
    y_input = np.random.uniform(height / 2, -height / 2, size=(input_points, 1))
    # aplha = np.repeat(np.random.random(1), input_points).reshape(input_points, 1)
    aplha = np.repeat(0.1745329, input_points).reshape(input_points, 1)  # 10 degree

    openfoam_invar_numpy = {"x": x_input, "y": y_input, "alpha": aplha}

    # add inferencer data
    grid_inference = PointwiseInferencer(
        nodes=nodes,
        invar=openfoam_invar_numpy,
        output_names=["u", "v", "phi"],
        batch_size=2048,
        plotter=PlotInference(),
    )
    flat_plate_domain.add_inferencer(grid_inference, "inf_data")


    # make solver
    slv = Solver(cfg, flat_plate_domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
