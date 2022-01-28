from sympy import Symbol, Eq, Abs
from sympy.logic.boolalg import Or
from modulus.solver import Solver
from modulus.dataset import TrainDomain, ValidationDomain
from modulus.data import Validation
from modulus.sympy_utils.functions import parabola
from modulus.sympy_utils.geometry_2d import Rectangle, Line
from modulus.csv_utils.csv_rw import csv_to_dict
from modulus.PDES.navier_stokes import NavierStokes, IntegralContinuity
from modulus.controller import ModulusController
import numpy as np

# params for domain
height = 0.59
width = 0.58
# inlet_vel = 10.0

# define geometry
rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))
# rec.rotate(4 * (np.pi / 180))
obstacle = Line((0, -height / 4), (0, height / 4), 1)
# I rotate the line by 90 degrees to make it horizontal.
obstacle.rotate(np.pi / 2)
geo = rec

plane1 = Line((-3 * width / 8, -height / 2), (-3 * width / 8, height / 2), 1)
plane2 = Line((-2 * width / 8, -height / 2), (-2 * width / 8, height / 2), 1)
plane3 = Line((-width / 8, -height / 2), (-width / 8, height / 2), 1)
plane4 = Line((0, -height / 2), (0, height / 2), 1)
plane5 = Line((width / 8, -height / 2), (width / 8, height / 2), 1)
plane6 = Line((2 * width / 8, -height / 2), (2 * width / 8, height / 2), 1)
plane7 = Line((3 * width / 8, -height / 2), (3 * width / 8, height / 2), 1)

# define sympy varaibles to parametize domain curves
x, y = Symbol("x"), Symbol("y")


class LDCTrain(TrainDomain):
    def __init__(self, **config):
        super(LDCTrain, self).__init__()

        # inlet
        # parabola_sympy = parabola(
        #     y, inter_1=-height / 2, inter_2=height / 2, height=inlet_vel
        # )
        # inletBC = geo.boundary_bc(
        #     outvar_sympy={"u": parabola_sympy, "v": 0},
        #     batch_size_per_area=64,
        #     criteria=Eq(x, -width / 2),
        # )

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

        inletBC = geo.boundary_bc(
            outvar_sympy={"u": 9.97564, "v": 0.697564},
            batch_size_per_area=64,
            criteria=Or(Eq(x, -width / 2), Eq(y, -height / 2)),
        )
        self.add(inletBC, name="Inlet")

        # outlet
        outletBC = geo.boundary_bc(
            outvar_sympy={"p": 0, "integral_continuity": 6.2774988},
            batch_size_per_area=64,
            criteria=Or(Eq(x, width / 2), Eq(y, height / 2)),
        )
        self.add(outletBC, name="Outlet")

        # topWall
        # topWall = geo.boundary_bc(
        #     outvar_sympy={"p": 0, "integral_continuity": 0.1333333},
        #     batch_size_per_area=256,
        #     criteria=Eq(y, height / 2),
        # )
        # self.add(topWall, name="TopWall")

        # bottomWall
        # bottomWall = geo.boundary_bc(
        #     outvar_sympy={"u": 9.97564, "v": 0.697564},
        #     batch_size_per_area=256,
        #     criteria=Eq(y, -height / 2),
        # )
        # self.add(bottomWall, name="BottomWall")

        # obstacleLine
        obstacleLine = obstacle.boundary_bc(
            outvar_sympy={"u": 0, "v": 0},
            batch_size_per_area=1000,
            lambda_sympy={"lambda_u": 10, "lambda_v": 10},
        )
        self.add(obstacleLine, name="obstacleLine")

        # interior
        interior = geo.interior_bc(
            outvar_sympy={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
            bounds={x: (-width / 2, width / 2), y: (-height / 2, height / 2)},
            lambda_sympy={
                "lambda_continuity": 10,
                "lambda_momentum_x": geo.sdf,
                "lambda_momentum_y": geo.sdf,
            },
            batch_size_per_area=10000,
        )
        self.add(interior, name="Interior")

        # planes
        plane1Cont = plane1.boundary_bc(
            outvar_sympy={"integral_continuity": 6.2774988},
            batch_size_per_area=256,
            lambda_sympy={"lambda_integral_continuity": 10},
        )
        plane2Cont = plane2.boundary_bc(
            outvar_sympy={"integral_continuity": 6.2774988},
            batch_size_per_area=256,
            lambda_sympy={"lambda_integral_continuity": 10},
        )
        plane3Cont = plane3.boundary_bc(
            outvar_sympy={"integral_continuity": 6.2774988},
            batch_size_per_area=256,
            lambda_sympy={"lambda_integral_continuity": 10},
        )
        plane4Cont = plane4.boundary_bc(
            outvar_sympy={"integral_continuity": 6.2774988},
            batch_size_per_area=256,
            lambda_sympy={"lambda_integral_continuity": 10},
        )
        plane5Cont = plane5.boundary_bc(
            outvar_sympy={"integral_continuity": 6.2774988},
            batch_size_per_area=256,
            lambda_sympy={"lambda_integral_continuity": 10},
        )
        plane6Cont = plane6.boundary_bc(
            outvar_sympy={"integral_continuity": 6.2774988},
            batch_size_per_area=256,
            lambda_sympy={"lambda_integral_continuity": 10},
        )
        plane7Cont = plane7.boundary_bc(
            outvar_sympy={"integral_continuity": 6.2774988},
            batch_size_per_area=256,
            lambda_sympy={"lambda_integral_continuity": 10},
        )

        self.add(plane1Cont, name="integralContinuity1")
        self.add(plane2Cont, name="integralContinuity2")
        self.add(plane3Cont, name="integralContinuity3")
        self.add(plane4Cont, name="integralContinuity4")
        self.add(plane5Cont, name="integralContinuity5")
        self.add(plane6Cont, name="integralContinuity6")
        self.add(plane7Cont, name="integralContinuity7")


# validation data
mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
openfoam_var = csv_to_dict("openfoam/cavity_uniformVel0.csv", mapping)
openfoam_var["x"] += -width / 2  # center OpenFoam data
openfoam_var["y"] += -height / 2  # center OpenFoam data
openfoam_invar_numpy = {
    key: value for key, value in openfoam_var.items() if key in ["x", "y"]
}
openfoam_outvar_numpy = {
    key: value for key, value in openfoam_var.items() if key in ["u", "v"]
}


class LDCVal(ValidationDomain):
    def __init__(self, **config):
        super(LDCVal, self).__init__()
        val = Validation.from_numpy(openfoam_invar_numpy, openfoam_outvar_numpy)
        self.add(val, name="Val")


class LDCSolver(Solver):
    train_domain = LDCTrain
    val_domain = LDCVal

    def __init__(self, **config):
        super(LDCSolver, self).__init__(**config)
        self.equations = (
            NavierStokes(nu=0.01, rho=1.0, dim=2, time=False).make_node()
            + IntegralContinuity().make_node()
        )
        flow_net = self.arch.make_node(
            name="flow_net", inputs=["x", "y"], outputs=["u", "v", "p"]
        )
        self.nets = [flow_net]

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update(
            {
                "network_dir": "./network_checkpoint_ldc_2d",
                "decay_steps": 4000,
                "max_steps": 400000,
            }
        )


if __name__ == "__main__":
    ctr = ModulusController(LDCSolver)
    ctr.run()