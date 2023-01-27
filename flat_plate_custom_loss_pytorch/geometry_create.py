from modulus.geometry.primitives_2d import Rectangle, Line
import numpy as np
from modulus.hydra import ModulusConfig


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

    height = cfg.custom.unscaled_domain_height * cfg.custom.obstacle_length  # Domain height
    width = cfg.custom.unscaled_domain_width * cfg.custom.obstacle_length  # Domain width

    geo = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))

    # creating flat plate
    obstacle_above = Line((0, 0), (0, cfg.custom.obstacle_length), 1).rotate(np.pi / 2)
    obstacle_below = Line((0, 0), (0, cfg.custom.obstacle_length), 1).rotate(np.pi / 2)

    # wk to enforce kutta condition 
    wk1_above = Line((0, -1 * cfg.custom.obstacle_length), (0, 0), 1).rotate(np.pi / 2)
    wk2_above = Line((0, -2 * cfg.custom.obstacle_length), (0, -1 * cfg.custom.obstacle_length),
                     1).rotate(np.pi / 2)
    wk3_above = Line((0, -3 * cfg.custom.obstacle_length), (0, -2 * cfg.custom.obstacle_length),
                     1).rotate(np.pi / 2)

    wk1_below = Line((0, -1 * cfg.custom.obstacle_length), (0, 0), 1).rotate(np.pi / 2)
    wk2_below = Line((0, -2 * cfg.custom.obstacle_length), (0, -1 * cfg.custom.obstacle_length),
                     1).rotate(np.pi / 2)
    wk3_below = Line((0, -3 * cfg.custom.obstacle_length), (0, -2 * cfg.custom.obstacle_length),
                     1).rotate(np.pi / 2)

    return geo, obstacle_above, obstacle_below, wk1_above, wk1_below, wk2_above, wk2_below, wk3_above, wk3_below
