from modulus.geometry.primitives_2d import Rectangle, Line
import numpy as np
from modulus.hydra import ModulusConfig
from modulus.geometry.parameterization import Parameterization
import sympy as sp


def create_geometry(cfg: ModulusConfig):
    """
    This function creates the required geometry (flat plate submerged in the flow field) and aligns as required.
    
    Parameters
    ----------
    cfg: It accesses the parameters set in the config file within config folder.

    Returns
    -------
    As it represents, it returns the geometry, flat plate and the wake region.
    """
    # make variables global
    alpha = sp.Symbol('alpha')
    height = cfg.custom.unscaled_domain_height * cfg.custom.obstacle_length  # Domain height
    width = cfg.custom.unscaled_domain_width * cfg.custom.obstacle_length  # Domain width

    # domain interior
    geo = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))

    # walls
    wall_left = Line((-width / 2, - height / 2), (- width / 2, height / 2), 1) #, parameterization=Parameterization({'alpha': 0.025}))
    wall_bottom = Line((- width / 2, - height / 2), (- width / 2, height / 2), 1).rotate(np.pi / 2)
    wall_right = Line((width / 2, - height / 2), (width / 2, height / 2), -1)
    wall_top = Line((width / 2, - height / 2), (width / 2, height / 2), -1).rotate(np.pi / 2)
    wall = [wall_left, wall_top, wall_right, wall_bottom]

    # creating flat plate
    obstacle_above = Line((0, 0), (0, cfg.custom.obstacle_length), 1).rotate(np.pi / 2)
    obstacle_below = Line((0, 0), (0, cfg.custom.obstacle_length), -1).rotate(np.pi / 2)
    obstacle = [obstacle_above, obstacle_below]

    # wk to enforce kutta condition 
    wk1_above = Line((0, -1 * cfg.custom.obstacle_length), (0, 0), 1).rotate(np.pi / 2)
    wk2_above = Line((0, -2 * cfg.custom.obstacle_length), (0, -1 * cfg.custom.obstacle_length), 1).rotate(np.pi / 2)
    wk3_above = Line((0, -3 * cfg.custom.obstacle_length), (0, -2 * cfg.custom.obstacle_length), 1).rotate(np.pi / 2)
    wk_above = [wk1_above, wk2_above, wk3_above]

    wk1_below = Line((0, -1 * cfg.custom.obstacle_length), (0, 0), -1).rotate(np.pi / 2)
    wk2_below = Line((0, -2 * cfg.custom.obstacle_length), (0, -1 * cfg.custom.obstacle_length), -1).rotate(np.pi / 2)
    wk3_below = Line((0, -3 * cfg.custom.obstacle_length), (0, -2 * cfg.custom.obstacle_length), -1).rotate(np.pi / 2)
    wk_below = [wk1_below, wk2_below, wk3_below]

    # IMP: all below quantities normals are changed from +1 to -1.
    # , obstacle_below, wk1_above, wk1_below, wk2_above, wk2_below, wk3_above, wk3_below,
    return geo, obstacle, wk_above, wk_below, wall


