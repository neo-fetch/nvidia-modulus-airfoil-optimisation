from modulus.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from modulus.domain import Domain
from modulus.hydra import ModulusConfig
from sympy import Symbol, Eq
import os
# misc
from cust_int_loss import PotentialLoss
import torch
import numpy as np


# from imp_measure import importance_measure


def define_domain(cfg: ModulusConfig, geo, obstacle, wk_above, wk_below, wall, alpha_range, nodes, height, width, u_x,
                  u_y, arch):
    x, y, alpha = Symbol('x'), Symbol('y'), Symbol('alpha')
    tot_cpus_avail = int(os.popen('nproc').read())

    alpha_one = lambda batch_size: np.full((batch_size, 1), np.random.uniform(- np.pi * cfg.custom.AoA / 180,
                                                            np.pi * cfg.custom.AoA / 180))

    ldc_domain = Domain()

    # Define Constraints
    leftWall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall[0],
        batch_size=cfg.batch_size.LeftWall,
        outvar={"u": u_x, "v": u_y},
        # criteria=~Eq(x, -width / 2),  # As the left wall lies on x = -width/2, we set the criteria to be x = -width/2
        # parameterization={'alpha': np.full((cfg.batch_size.LeftWall, 1), np.random.uniform(- 10 / 180, 10 / 180))},
        # Unhashed error if used another dictionary: {alpha_range} instead use alpha_range or remove {}.
        parameterization=alpha_range,
        fixed_dataset=True,
        batch_per_epoch=1,
        quasirandom=True,
        num_workers=tot_cpus_avail,
    )
    # Add constraints to solver
    ldc_domain.add_constraint(leftWall, name="LeftWall")


    # copy_alpha_1 = lambda batch_size, alpha_same: np.full((batch_size, 1), alpha_same)
    # alpha_same = ldc_domain.constraints['LeftWall'].dataset.invar['alpha'][0]
    # copy_alpha = lambda batch_size: np.full((batch_size, 1), alpha_same)
    topWall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall[1],
        batch_size=cfg.batch_size.TopWall,
        outvar={"u": u_x, "v": u_y},
        # Mimicing the far field conditions "u":u_x , "v": u_y,
        # criteria=~Eq(y, height / 2),  # As the top wall lies on y = height/2, we set the criteria to be y = height/2
        # parameterization=alpha_range,
        # parameterization={'alpha': copy_alpha_1(batch_size=cfg.batch_size.TopWall, alpha_same=ldc_domain.constraints['LeftWall'].dataset.invar['alpha'][0])},
        parameterization=alpha_range,
        fixed_dataset=True,
        batch_per_epoch=1,
        quasirandom=True,
        num_workers=tot_cpus_avail,
    )
    ldc_domain.add_constraint(topWall, name="TopWall")

    rightWall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall[2],
        batch_size=cfg.batch_size.RightWall,
        outvar={"u": u_x, "v": u_y},
        # Mimicing the far field conditions "u":u_x , "v": u_y,
        # criteria=~Eq(x, width / 2),  # As the right wall lies on x = width/2, we set the criteria to be x = width/2
        # parameterization={'alpha': copy_alpha},
        parameterization=alpha_range,
        fixed_dataset=True,
        batch_per_epoch=1,
        quasirandom=True,
        num_workers=tot_cpus_avail,
    )
    ldc_domain.add_constraint(rightWall, name="RightWall")

    bottomWall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall[3],
        batch_size=cfg.batch_size.BottomWall,
        outvar={"u": u_x, "v": u_y},
        # Mimicing the far field conditions "u":u_x , "v": u_y,
        # criteria=~Eq(y, -height / 2),
        # As the bottom wall lies on y = -height/2, we set the criteria to be y = -height/2
        # parameterization={'alpha': copy_alpha},
        parameterization=alpha_range,
        fixed_dataset=True,
        batch_per_epoch=1,
        quasirandom=True,
        num_workers=tot_cpus_avail,
    )
    ldc_domain.add_constraint(bottomWall, name="BottomWall")

    obstacleLineAbove = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=obstacle[0],
        batch_size=cfg.batch_size.obstacle_above,
        outvar={"residual_plate_above": 0, "v": 0},
        # Setting up the no slip condition for the obstacle.
        # lambda_weighting={"residual_u": 100, "residual_obstacle_above": 100},  # Symbol("sdf")},
        # check Symbol("sdf") --> geo.sdf # Weights for the loss function.
        # parameterization={'alpha': copy_alpha},
        parameterization=alpha_range,
        fixed_dataset=True,
        batch_per_epoch=1,
        quasirandom=True,
        num_workers=tot_cpus_avail,
    )
    ldc_domain.add_constraint(obstacleLineAbove, name="obstacleLineAbove")

    obstacleLineBelow = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=obstacle[1],
        batch_size=cfg.batch_size.obstacle_below,
        outvar={"u": u_x, "v": 0},
        # lambda_weighting={"u": 100, "residual_obstacle_below": 100},
        # parameterization={'alpha': copy_alpha},
        parameterization=alpha_range,
        fixed_dataset=True,
        batch_per_epoch=1,
        quasirandom=True,
        num_workers=tot_cpus_avail,
    )
    ldc_domain.add_constraint(obstacleLineBelow, name="obstacleLineBelow")

    l = lambda x: x / (3 * cfg.custom.obstacle_length)  # x = 0 at the trailing edge of the obstacle

    wakeLine1_Above = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wk_above[0],
        batch_size=cfg.batch_size.wake1_above,
        outvar={"u": u_x, "v": u_y * l(x)},
        # lambda_weighting={"u": 100, "v": 100},
        # parameterization={'alpha': copy_alpha},
        parameterization=alpha_range,
        fixed_dataset=True,
        batch_per_epoch=1,
        quasirandom=True,
        num_workers=tot_cpus_avail,
    )
    ldc_domain.add_constraint(wakeLine1_Above, name="wakeLine1_Above")

    wakeLine2_Above = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wk_above[1],
        batch_size=cfg.batch_size.wake2_above,
        outvar={"u": u_x, "v": u_y * l(x)},
        # lambda_weighting={"u": 100, "v": 100},
        # parameterization={'alpha': copy_alpha},
        parameterization=alpha_range,
        fixed_dataset=True,
        batch_per_epoch=1,
        quasirandom=True,
        num_workers=tot_cpus_avail,
    )
    ldc_domain.add_constraint(wakeLine2_Above, name="wakeLine2_Above")

    wakeLine3_Above = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wk_above[2],
        batch_size=cfg.batch_size.wake3_above,
        outvar={"u": u_x, "v": u_y * l(x)},
        # lambda_weighting={"u": 100, "v": 100},
        # parameterization={'alpha': copy_alpha},
        parameterization=alpha_range,
        fixed_dataset=True,
        batch_per_epoch=1,
        quasirandom=True,
        num_workers=tot_cpus_avail,
    )

    ldc_domain.add_constraint(wakeLine3_Above, name="wakeLine3_Above")

    wakeLine1_Below = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wk_below[0],
        batch_size=cfg.batch_size.wake1_below,
        outvar={"u": u_x, "v": u_y * l(x)},
        # lambda_weighting={"u": 100, "v": 100},
        # parameterization={'alpha': copy_alpha},
        parameterization=alpha_range,
        fixed_dataset=True,
        batch_per_epoch=1,
        quasirandom=True,
        num_workers=tot_cpus_avail,
    )
    ldc_domain.add_constraint(wakeLine1_Below, name="wakeLine1_Below")

    wakeLine2_Below = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wk_below[1],
        outvar={"u": u_x, "v": u_y * l(x)},
        batch_size=cfg.batch_size.wake2_below,  # batch_size=150 * 2
        # lambda_weighting={"u": 100, "v": 100},
        # parameterization={'alpha': copy_alpha},
        parameterization=alpha_range,
        fixed_dataset=True,
        batch_per_epoch=1,
        quasirandom=True,
        num_workers=tot_cpus_avail,
    )
    ldc_domain.add_constraint(wakeLine2_Below, name="wakeLine2_Below")

    wakeLine3_Below = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wk_below[2],
        outvar={"u": u_x, "v": u_y * l(x)},
        batch_size=cfg.batch_size.wake1_above,  # batch_size=150 * 2
        # lambda_weighting={"u": 100, "v": 100},
        # parameterization={'alpha': copy_alpha},
        parameterization=alpha_range,
        fixed_dataset=True,
        batch_per_epoch=1,
        quasirandom=True,
        num_workers=tot_cpus_avail,
    )
    ldc_domain.add_constraint(wakeLine3_Below, name="wakeLine3_Below")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # importance_model_graph = Graph(nodes, invar=[Key("x"), Key("y")], req_names=[
    #     Key("u", derivatives=[Key("x")]),
    #     Key("u", derivatives=[Key("y")]),
    #     Key("v", derivatives=[Key("x")]),
    #     Key("v", derivatives=[Key("y")]),
    # ],
    #                                ).to(device)

    def importance_measure(invar):
        # outvar = importance_model_graph(
        #     Constraint._set_device(invar, device=device, requires_grad=True)
        # )
        importance = None
        # return importance.cpu().detach().numpy()
        return importance

    interior = PointwiseInteriorConstraint(
        geometry=geo,
        nodes=nodes,
        outvar={"Poisson_2D": 0, "residual_u": 0, "residual_v": 0},
        bounds=geo.bounds.bound_ranges,
        # lambda_weighting={
        #    "Poisson_2D": Symbol("sdf"),
        #    "residual_u": Symbol("sdf"),
        #    "residual_v": Symbol("sdf"),
        # },
        # parameterization={'alpha': copy_alpha},
        parameterization=alpha_range,
        fixed_dataset=True,
        batch_per_epoch=1,
        quasirandom=True,
        # importance_measure=importance_measure,
        loss=PotentialLoss(ldc_domain, cfg, arch, ldc_domain.constraints['LeftWall'].dataset.invar['alpha'][0]),
        batch_size=cfg.batch_size.Interior,
        num_workers=tot_cpus_avail,
    )

    ldc_domain.add_constraint(interior, name="interior")

    return ldc_domain
