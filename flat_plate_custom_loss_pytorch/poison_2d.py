from modulus.eq.pde import PDE
from sympy import Symbol, Function
from sympy import sin, cos
from modulus.hydra import ModulusConfig
import numpy as np


class Poison_2D(PDE):
    name = "Poisson_2D"

    def __init__(self, cfg: ModulusConfig):
        # super().__init__()
        x, y, alpha = Symbol("x"), Symbol("y"), Symbol("alpha")
        input_variables = {"x": x, "y": y, "alpha": alpha}

        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        phi = Function("phi")(*input_variables)

        u_comp = cfg.custom.free_stream_velocity * cos(alpha)

        # For the far field conditions, we need to define the boundary conditions for the velocity components
        self.equations = {
                          "residual_plate_above": u - phi.diff(x) - u_comp,
                          # "residual_u_comp": u - cfg.custom.free_stream_velocity * cos(alpha),  # np.abs
                          # "residual_v_comp": v - cfg.custom.free_stream_velocity * sin(alpha),  # np.abs
                          # "residual_obstacle_above": v,
                          # "residual_obstacle_below": v,
                          # "residual_obstacle_wake1_above": v - cfg.custom.free_stream_velocity * sin(alpha) * x / (
                          #         3 * cfg.custom.obstacle_length),
                          # "residual_obstacle_wake2_above": v - cfg.custom.free_stream_velocity * sin(alpha) * x / (
                          #         3 * cfg.custom.obstacle_length),
                          # "residual_obstacle_wake3_above": v - cfg.custom.free_stream_velocity * sin(alpha) * x / (
                          #         3 * cfg.custom.obstacle_length),
                          # "residual_obstacle_wake1_below": v - cfg.custom.free_stream_velocity * sin(alpha) * x / (
                          #         3 * cfg.custom.obstacle_length),
                          # "residual_obstacle_wake2_below": v - cfg.custom.free_stream_velocity * sin(alpha) * x / (
                          #         3 * cfg.custom.obstacle_length),
                          # "residual_obstacle_wake3_below": v - cfg.custom.free_stream_velocity * sin(alpha) * x / (
                          #         3 * cfg.custom.obstacle_length),
                          "residual_u": u - phi.diff(x),
                          "residual_v": v - phi.diff(y),
                          "Poisson_2D": (phi.diff(x)).diff(x) + (phi.diff(y)).diff(y)
                          }
