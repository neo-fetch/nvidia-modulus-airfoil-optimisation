from sympy import Symbol, Eq, Abs

from modulus.solver import Solver
from modulus.dataset import TrainDomain, ValidationDomain
from modulus.data import Validation
from modulus.sympy_utils.functions import parabola
from modulus.sympy_utils.geometry_2d import Rectangle, Line
from modulus.csv_utils.csv_rw import csv_to_dict
from modulus.PDES.navier_stokes import NavierStokes, IntegralContinuity
from modulus.controller import ModulusController

# params for domain
height = 0.1
width = 0.1
inlet_vel = 1.0

# define geometry
rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))
obstacle = Line((0, -height/4), (0, height/4), 1)
obstacle.rotate(1.05)
geo = rec

# define sympy varaibles to parametize domain curves
x, y, alpha = Symbol('x'), Symbol('y'), Symbol('alpha')
param_ranges = {alpha: (-0.1744, 0.1744)}
fixed_param_range = {alpha: lambda batch_size: np.full((batch_size, 1), np.random.uniform(-0.1744, 0.1744))}

class LDCTrain(TrainDomain):
  def __init__(self, **config):
    super(LDCTrain, self).__init__()

    #inlet
    u_x = 10*cos(alpha)
    flow_rate = u_x*width
    inletBC = geo.boundary_bc(outvar_sympy={'u': u_x, 'v': 0},
                              batch_size_per_area=64,
                              criteria=Eq(x, -width/2),
                              param_ranges={**fixed_param_range},
                              fxied_var=False)
    self.add(inletBC, name="Inlet")

    #outlet
    outletBC = geo.boundary_bc(outvar_sympy={'p': 0, 'integral_continuity': 1},
                               batch_size_per_area=64,
                               criteria=Eq(x, width/2),
                               param_ranges=param_ranges)
    self.add(outletBC, name="Outlet")

    #topWall
    topWall = geo.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                              batch_size_per_area=256,
                              criteria=Eq(y, height/2),
                              param_ranges=param_ranges)
    self.add(topWall, name="TopWall")

    #bottomWall
    bottomWall = geo.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                 batch_size_per_area=256,
                                 criteria=Eq(y, -height/2))
    self.add(bottomWall, name="BottomWall")

    #obstacleLine
    obstacleLine = obstacle.boundary_bc(outvar_sympy = {'u': 0, 'v': 0},
                                   batch_size_per_area=1000,
                                   lambda_sympy={'lambda_u':10, 'lambda_v':10},
                                   param_ranges=param_ranges)
    self.add(obstacleLine, name="obstacleLine")

    # interior
    interior = geo.interior_bc(outvar_sympy={'continuity': 0, 'momentum_x': 0, 'momentum_y': 0},
                               bounds={x: (-width / 2, width / 2),
                                       y: (-height / 2, height / 2)},
                               lambda_sympy={'lambda_continuity': 10,
                                             'lambda_momentum_x': geo.sdf,
                                             'lambda_momentum_y': geo.sdf},
                               batch_size_per_area=10000,
                               param_ranges=param_ranges)
    self.add(interior, name="Interior")

    #planes

# validation data
mapping = {'Points:0': 'x', 'Points:1': 'y', 'U:0': 'u', 'U:1': 'v', 'p': 'p'}
openfoam_var = csv_to_dict('openfoam/cavity_uniformVel0.csv', mapping)
openfoam_var['x'] += -width / 2  # center OpenFoam data
openfoam_var['y'] += -height / 2  # center OpenFoam data
openfoam_invar_numpy = {key: value for key, value in openfoam_var.items() if key in ['x', 'y']}
openfoam_outvar_numpy = {key: value for key, value in openfoam_var.items() if key in ['u', 'v']}

class LDCVal(ValidationDomain):
  def __init__(self, **config):
    super(LDCVal, self).__init__()
    val = Validation.from_numpy(openfoam_invar_numpy, openfoam_outvar_numpy)
    self.add(val, name='Val')

class LDCSolver(Solver):
  train_domain = LDCTrain
  val_domain = LDCVal

  def __init__(self, **config):
    super(LDCSolver, self).__init__(**config)
    self.equations = (NavierStokes(nu=0.01, rho=1.0, dim=2,time=False).make_node()
                      +IntegralContinuity().make_node())
    flow_net = self.arch.make_node(name='flow_net',
                                   inputs=['x', 'y', 'alpha'],
                                   outputs=['u', 'v', 'p'])
    self.nets = [flow_net]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'network_dir': './network_checkpoint_ldc_2d',
        'decay_steps': 4000,
        'max_steps': 400000
    })


if __name__ == '__main__':
  ctr = ModulusController(LDCSolver)
  ctr.run()
