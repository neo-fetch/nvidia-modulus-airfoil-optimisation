from sympy import Symbol, Eq, Ge, Abs, Function, Number, sin, cos
from modulus.pdes import PDES
from modulus.variables import Variables
import time
from modulus.solver import Solver
from modulus.dataset import TrainDomain, InferenceDomain
from modulus.data import Inference
from modulus.sympy_utils.geometry_2d import Rectangle, Line
from modulus.controller import ModulusController
import numpy as np
import math
import sys

def get_angle(theta, magnitude):
    return math.cos(theta)*magnitude, math.sin(theta)*magnitude

class Poisson_2D(PDES):
    name = 'Poisson_2D'
    def __init__(self):
        # coordinates
        x, y = Symbol('x'), Symbol('y')

        # angle of attack
        alpha = Symbol('alpha')

        # make input variables
        input_variables = {'x': x, 'y': y, 'alpha': alpha}

        obstacle_length = 0.10
        # potential
        phi = Function('phi')(*input_variables)
        u = Function('u')(*input_variables)
        v = Function('v')(*input_variables)
        
        self.equations = Variables()
        # Here I implement a simpler form of a 2D Navier-Stokes equation in the form of laplacian(u,v) = 0 such that
        # laplacian(u,v).diff(u) = u and laplacian(u,v).diff(v) = v
        # laplacian(u,v).diff(t) = 0
        # self.equations['u'] = phi.diff(x) 
        # self.equations['v'] = phi.diff(y) # redefined below to facilitate tensorboard graphs.
        self.equations['residual_u'] = u - phi.diff(x)
        self.equations['residual_v'] = v - phi.diff(y)
        # For the far field conditions, we need to define the boundary conditions for the velocity components
        self.equations['residual_u_comp'] = u - 10*cos(alpha)
        self.equations['residual_v_comp'] = v - 10*sin(alpha)
        # For the obstacle inside geometry, v = 0 because v = V(perturbation) + V(far field)) is 0 
        self.equations['residual_obstacle_above'] = v
        self.equations['residual_obstacle_below'] = v
        # We divide the wake into three parts: the first part after the trailing edge, then the second and then finally, the third part.
        # This is done to observe how the error manifests in each of the three parts.
        self.equations['residual_obstacle_wake1_above'] = v # - 10*sin(alpha)*(x)/(3*obstacle_length)
        self.equations['residual_obstacle_wake2_above'] = v # - 10*sin(alpha)*(x)/(3*obstacle_length)
        self.equations['residual_obstacle_wake3_above'] = v # - 10*sin(alpha)*(x)/(3*obstacle_length)

        self.equations['residual_obstacle_wake1_below'] = v # - 10*sin(alpha)*(x)/(3*obstacle_length)
        self.equations['residual_obstacle_wake2_below'] = v # - 10*sin(alpha)*(x)/(3*obstacle_length)
        self.equations['residual_obstacle_wake3_below'] = v # - 10*sin(alpha)*(x)/(3*obstacle_length)

        # The Laplacian we are going to solve is:
        self.equations['Poisson_2D'] = (phi.diff(x)).diff(x) + (phi.diff(y)).diff(y) # grad^2(phi)


# params for domain
obstacle_length = 0.10
height = 6*obstacle_length  
width = 6*obstacle_length

# define geometry
rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))
obstacle_above = Line((0, 0), (0, obstacle_length), 1)
obstacle_below = Line((0, 0), (0, obstacle_length), 1)

wake1_above = Line((0, -1*obstacle_length), (0, 0), 1) # Wake to enforce kutta condition
wake2_above = Line((0, -2*obstacle_length), (0, -1*obstacle_length), 1) # Wake to enforce kutta condition
wake3_above = Line((0, -3*obstacle_length), (0, -2*obstacle_length), 1) # Wake to enforce kutta condition

wake1_below = Line((0, -1*obstacle_length), (0, 0), 1) # Wake to enforce kutta condition
wake2_below = Line((0, -2*obstacle_length), (0, -1*obstacle_length), 1) # Wake to enforce kutta condition
wake3_below = Line((0, -3*obstacle_length), (0, -2*obstacle_length), 1) # Wake to enforce kutta condition

obstacle_above.rotate(np.pi / 2)
obstacle_below.rotate(np.pi / 2)

wake1_above.rotate(np.pi / 2)
wake2_above.rotate(np.pi / 2)
wake3_above.rotate(np.pi / 2)

wake1_below.rotate(np.pi / 2)
wake2_below.rotate(np.pi / 2)
wake3_below.rotate(np.pi / 2)

# I rotate the line by 90 degrees to make it horizontal. 
# Now, the way this system is set up, the line will be positioned such that it is two units from the left of the rectangle, and 3 units 
# from its trailing edge. 

geo = rec

# define sympy varaibles to parametize domain curves
x, y, alpha = Symbol('x'), Symbol('y'), Symbol('alpha')
# limit the range of alpha from -10 to 10 using np.pi.
param_ranges_above = {alpha: lambda batch_size: np.full((batch_size, 1), np.random.uniform(-np.pi*10/180, np.pi*10/180)), y: lambda batch_size: np.full((batch_size, 1), np.random.uniform(height/2, height))}
param_ranges_below = {alpha: lambda batch_size: np.full((batch_size, 1), np.random.uniform(-np.pi*10/180, np.pi*10/180)), y: lambda batch_size: np.full((batch_size, 1), np.random.uniform(0, height/2))}
fixed_param_range = {alpha: lambda batch_size: np.full((batch_size, 1), np.random.uniform(-np.pi*10/180, np.pi*10/180))}

class PotentialTrain(TrainDomain):
    def __init__(self, **config):
        super(PotentialTrain, self).__init__()

#############################################################################################
        # I want to make the inlet velocity to be 10.0 m/s with an incidence angle of 4 degrees at the obstacle.
        # the inverse tan(v/u) gives me the required angle of incidence.
        # Drawing the scenario in comments below:
        #      +---------+
        #     /|/     \|/|
        #    //|// --- //|
        #   ///|/////////|
        #  ////+---------+
        #  //////////////
        #  / ////////////
        #    / //////////
        # where / is u + v such that tan-1(v/u) = x degrees(here i kept x as 4).
        u_x = 10*cos(alpha)
        u_y = 10*sin(alpha)
        flow_rate = u_x*width + u_y*height

        # inlet
        inletBC = geo.boundary_bc(
            outvar_sympy={"residual_u_comp": 0, "residual_v_comp": 0}, # "u": u_x, "v": u_y, 
            batch_size_per_area=250*2,
            criteria=Eq(x, -width / 2),
            param_ranges ={**fixed_param_range},
            fixed_var=False
        )
        self.add(inletBC, name="Inlet")

        # outlet
        outletBC = geo.boundary_bc(
            outvar_sympy={"residual_u_comp": 0, "residual_v_comp": 0}, # Mimicing the far field conditions "u":u_x , "v": u_y, 
            batch_size_per_area=500*2,
            criteria=Ge(y/height+x/width, 1/2),
            param_ranges ={**fixed_param_range},
            fixed_var=False
        )
        self.add(outletBC, name="Outlet")

        # bottomWall
        bottomWall = geo.boundary_bc(
            outvar_sympy={"residual_u_comp": 0, "residual_v_comp": 0}, # "u": u_x, "v": u_y
            batch_size_per_area=250*2,
            criteria=Eq(y, -height / 2),
            param_ranges ={**fixed_param_range},
            fixed_var=False            
        )
        self.add(bottomWall, name="BottomWall")

        # obstacleLine Above
        obstacleLineAbove = obstacle_above.boundary_bc(
            outvar_sympy={"u": u_x, 'residual_obstacle_above': 0},
            batch_size_per_area=600*2,
            lambda_sympy={"lambda_u": 100, "lambda_v": 100, "lambda_residual_obstacle_above": geo.sdf},
            param_ranges ={**param_ranges_above},
            fixed_var=False            
        )
        self.add(obstacleLineAbove, name="obstacleLineAbove")

        # obstacleLine Below
        obstacleLineBelow = obstacle_below.boundary_bc(
            outvar_sympy={"u": u_x, 'residual_obstacle_below': 0},
            batch_size_per_area=600*2,
            lambda_sympy={"lambda_u": 100, "lambda_v": 100, "lambda_residual_obstacle_below": geo.sdf},
            param_ranges ={**param_ranges_below},
            fixed_var=False
        )
        self.add(obstacleLineBelow, name="obstacleLineBelow")

        # wakeLine
        # Here we define u = u and v = 0 at the trailing edge of the obstacle(which is at x=0, and v = v at x = right wall). As a linear function for simplicity.
        # As the trailing edge is positioned at {0, 0}, we see the 
        l = lambda x : (x)/(3*obstacle_length) # x = 0 at the trailing edge of the obstacle
        wakeLine1_Above = wake1_above.boundary_bc(
            outvar_sympy={"u": u_x, "v": u_y*l(x), 'residual_obstacle_wake1_above': 0},
            batch_size_per_area=150*2,
            lambda_sympy={"lambda_u": 100, "lambda_v": 100, "lambda_residual_obstacle_wake1_above": geo.sdf},
            param_ranges ={**param_ranges_above},
            fixed_var=False            
        )
        self.add(wakeLine1_Above, name="wakeLine1_Above")

        wakeLine2_Above = wake2_above.boundary_bc(
            outvar_sympy={"u": u_x, "v": u_y*l(x), 'residual_obstacle_wake2_above': 0},
            batch_size_per_area=150*2,
            lambda_sympy={"lambda_u": 100, "lambda_v": 100, "lambda_residual_obstacle_wake2_above": geo.sdf},
            param_ranges ={**param_ranges_above},
            fixed_var=False
        )
        self.add(wakeLine2_Above, name="wakeLine2_Above")

        wakeLine3_Above = wake3_above.boundary_bc(
            outvar_sympy={"u": u_x, "v": u_y*l(x), 'residual_obstacle_wake3_above': 0},
            batch_size_per_area=150*2,
            lambda_sympy={"lambda_u": 100, "lambda_v": 100, "lambda_residual_obstacle_wake3_above": geo.sdf},
            param_ranges ={**param_ranges_above},
            fixed_var=False
        )

        self.add(wakeLine3_Above, name="wakeLine3_Above")

        wakeLine1_Below = wake1_below.boundary_bc(
            outvar_sympy={"u": u_x, "v": u_y*l(x), 'residual_obstacle_wake1_below': 0},
            batch_size_per_area=150*2,
            lambda_sympy={"lambda_u": 100, "lambda_v": 100, "lambda_residual_obstacle_wake1_below": geo.sdf},
            param_ranges ={**param_ranges_below},
            fixed_var=False
        )
        self.add(wakeLine1_Below, name="wakeLine1_Below")

        wakeLine2_Below = wake2_below.boundary_bc(
            outvar_sympy={"u": u_x, "v": u_y*l(x), 'residual_obstacle_wake2_below': 0},
            batch_size_per_area=150*2,
            lambda_sympy={"lambda_u": 100, "lambda_v": 100, "lambda_residual_obstacle_wake2_below": geo.sdf},
            param_ranges ={**param_ranges_below},
            fixed_var=False
        )
        self.add(wakeLine2_Below, name="wakeLine2_Below")

        wakeLine3_Below = wake3_below.boundary_bc(
            outvar_sympy={"u": u_x, "v": u_y*l(x), 'residual_obstacle_wake3_below': 0},
            batch_size_per_area=150*2,
            lambda_sympy={"lambda_u": 100, "lambda_v": 100, "lambda_residual_obstacle_wake3_below": geo.sdf},
            param_ranges ={**param_ranges_below},
            fixed_var=False
        )
        self.add(wakeLine3_Below, name="wakeLine3_Below")

        # interior
        interior = geo.interior_bc(
            outvar_sympy={"Poisson_2D": 0, "residual_u": 0, "residual_v": 0},
            bounds={x: (-width / 2, width / 2), y: (-height / 2, height / 2)},
            lambda_sympy={
                "lambda_Poisson_2D": geo.sdf,
                "lambda_residual_u": geo.sdf,
                "lambda_residual_v": geo.sdf,
            },
            batch_size_per_area=2000*2,
            param_ranges ={**fixed_param_range},
            fixed_var=False            
        )
        self.add(interior, name="Interior")

        neighbourhood = geo.interior_bc(
            outvar_sympy={"Poisson_2D": 0, "residual_u": 0, "residual_v": 0},
            bounds={x: (-height / 3, height / 3), y: (-height / 8, height / 8)},
            lambda_sympy={
                "lambda_Poisson_2D": geo.sdf,
                "lambda_residual_u": geo.sdf,
                "lambda_residual_v": geo.sdf,
            },
            batch_size_per_area=2000*2,
            param_ranges ={**fixed_param_range},
            fixed_var=False            
        )
        self.add(neighbourhood, name="Neighbourhood")

class PotentialInference(InferenceDomain):
    def __init__(self, **config):
        super(PotentialInference, self).__init__()
        x, y, alpha = Symbol('x'), Symbol('y'), Symbol('alpha')
        interior = Inference(geo.sample_interior(10000, bounds={x: (-width / 2, width / 2), y: (-height / 2, height / 2)}, param_ranges={alpha: np.pi*(10/180)}), ['u', 'v', 'phi'])
        self.add(interior, name="Inference")

class PotentialSolver(Solver):
    train_domain = PotentialTrain
    inference_domain = PotentialInference

    def __init__(self, **config):
        super(PotentialSolver, self).__init__(**config)
        self.equations = (
            Poisson_2D().make_node()
        )
        flow_net = self.arch.make_node(
            name="flow_net", inputs=["x", "y", "alpha"], outputs=["u", "v", "phi"]
        )
        self.nets = [flow_net]


    @classmethod
    def update_defaults(cls, defaults):
        defaults.update(
            {
                "network_dir": "./network_checkpoint_potential_flow_2d",
                "decay_steps": 4000,
                "max_steps": 400000,
                "layer_size": 100,
            }
        )

if __name__ == "__main__":
    ctr = ModulusController(PotentialSolver)
    ctr.run()
