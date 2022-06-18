# Importing relevant libraries
from sympy import Symbol, Eq, Ge, Abs, Function, Number, sin, cos
from modulus.pdes import PDES
from modulus.variables import Variables
from modulus.solver import Solver
from modulus.dataset import TrainDomain, ValidationDomain, MonitorDomain, InferenceDomain
from modulus.data import Validation, Monitor, Inference
import tensorflow as tf
from modulus.sympy_utils.geometry_2d import Rectangle, Line
from modulus.sympy_utils.geometry_1d import Line1D
from modulus.controller import ModulusController
import numpy as np
import sys


# k = int(input("Enter your 'k': "))

class Sin2kx(PDES):
    name = 'Sin2kx'
    def __init__(self):
        # coordinates
        x = Symbol('x')
        # make input variables
        input_variables = {'x': x}

        # potential
        f = Function('f')(*input_variables)
        
        self.equations = Variables()
        self.equations['Sin2kx'] = -f + sin(2*x)/2 + sin(4*x)/4 + sin(6*x)/6 + sin(8*x)/8 + sin(10*x)/10
        # self.equations['Sin2kx'] = -f.diff(x, 1) + cos(2*x) + cos(4*x) + cos(6*x) + cos(8*x) + cos(10*x)
        # self.equations['Sin2kx'] = f.diff(x, 2) + 2*sin(2*x) + 4*sin(4*x) + 6*sin(6*x) + 8*sin(8*x) + 10*sin(10*x)

        # self.equations['Sin2kx'] = f.diff(x, 3) + 2*2*cos(2*x) + 4*4*cos(4*x) + 6*6*cos(6*x) + 8*8*cos(8*x) + 10*10*cos(10*x)
        self.equations['Sin2kx'] = -f.diff(x, 4) + 2*2*2*sin(2*x) + 4*4*4*sin(4*x) + 6*6*6*sin(6*x) + 8*8*8*sin(8*x) + 10*10*10*sin(10*x)
        # self.equations['diff2_f'] = f.diff(x, 2) + 2*sin(2*x) + 4*sin(4*x) + 6*sin(6*x) + 8*sin(8*x) + 10*sin(10*x)
        # self.equations['diff_f'] = f.diff(x, 1) 
# ---------------------------------------------------------------------------------------------------------------------- #

# params for domain
L = float(np.pi)
geo = Line1D(-L, L)
x = Symbol('x')

class PotentialTrain(TrainDomain):
    def __init__(self, **config):
        super(PotentialTrain, self).__init__()

        end_points = geo.boundary_bc(
            outvar_sympy={"f": 0, 'f__x': 5}, # Mimicing the far field conditions "u":u_x , "v": u_y,
            batch_size_per_area=200,
            lambda_sympy={"lambda_f": 0.25, "lambda_f__x": 0.25}, # Weights for the loss function.
        )
        self.add(end_points, name="end_points")

        # interior
        interior = geo.interior_bc(
            outvar_sympy={"Sin2kx": 0},
            bounds={x: (-L, L)},
            lambda_sympy={
                "lambda_Sin2kx": geo.sdf,
            },
            
            batch_size_per_area=2000,
        )
        self.add(interior, name="interior")

        # neighborhood = geo.interior_bc(
        #     outvar_sympy={"Sin2kx": 0, "residual_u": 0, "residual_v": 0},
        #     bounds={x: (-height / 3, height / 3), y: (-height / 8, height / 8)},
        #     lambda_sympy={
        #         "lambda_Sin2kx": geo.sdf,
        #         "lambda_residual_u": geo.sdf,
        #         "lambda_residual_v": geo.sdf,
        #     },
        #     batch_size_per_area=2000*2,
        #     param_ranges ={**fixed_param_range},
        #     fixed_var=False            
        # )
        # self.add(neighborhood, name="neighborhood")

# ---------------------------------------------------------------------------------------------------------------------- #

# We will now see how to create an Inference domain to plot the desired variables in the interior
# at a desired point density.
# The LDCInference class can be created by inheriting from the InferenceDomain parent class. The points are again
# sampled in a similar way as done during the definition of LDCTrain and LDCMonitor domains.

class PotentialInference(InferenceDomain):
    def __init__(self, **config):
        super(PotentialInference, self).__init__()
        x = Symbol('x')
        interior = Inference(geo.sample_interior(1e05, bounds={x: (-L, L)}), ['f'])
        self.add(interior, name="Inference")

# ---------------------------------------------------------------------------------------------------------------------- #

class PotentialSolver(Solver):
    train_domain = PotentialTrain
    inference_domain = PotentialInference

    def __init__(self, **config):
        super(PotentialSolver, self).__init__(**config)
        self.equations = (
            Sin2kx().make_node()
        )
        flow_net = self.arch.make_node(
            name="flow_net", inputs=["x"], outputs=["f"]
        )
        self.nets = [flow_net]

    # def custom_loss(self, domain_invar, pred_domain_outvar, true_domain_outvar, step):
    #     x_interior_1 = domain_invar['interior']['x']
    #     f_result = self.nets[0].evaluate({'x': x_interior_1[0]})
    #     f_diff1 = tf.gradients(f_result, x_interior_1)[0]
    #     print(f_diff1)

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update(
            {
                "network_dir": "./network_checkpoint_sine_k_all_5",
                "decay_steps": 4000,
                "max_steps": 400000,
                "layer_size": 100,
            }
        )

# ---------------------------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    ctr = ModulusController(PotentialSolver)
    ctr.run()

