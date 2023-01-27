from modulus.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from modulus.domain import Domain
from modulus.hydra import ModulusConfig
from sympy import Symbol, Eq

#misc
from cust_int_loss import PotentialLoss


def define_domain(cfg: ModulusConfig, geo, obstacle_above, obstacle_below, wake1_above, wake1_below, wake2_above,
                  wake2_below, wake3_above, wake3_below, alpha_range, nodes, height, width, u_x, u_y, arch):
    x, y, alpha = Symbol('x'), Symbol('y'), Symbol('alpha')
    ldc_domain = Domain()

    # Define Constraints
    leftWall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        batch_size=cfg.batch_size.LeftWall,
        outvar={"residual_u_comp": 0, "residual_v_comp": 0},
        criteria=~Eq(x, -width / 2),  # As the left wall lies on x = -width/2, we set the criteria to be x = -width/2
        parameterization=alpha_range,
        # Unhashed error if used another dictionary: {alpha_range} instead use alpha_range or remove {}.
        fixed_dataset=True,
        batch_per_epoch=1,
        quasirandom=False,
    )
    # Add constraints to solver
    ldc_domain.add_constraint(leftWall, name="LeftWall")

    topWall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        batch_size=cfg.batch_size.TopWall,
        outvar={"residual_u_comp": 0, "residual_v_comp": 0},
        # Mimicing the far field conditions "u":u_x , "v": u_y,
        criteria=~Eq(y, height / 2),  # As the top wall lies on y = height/2, we set the criteria to be y = height/2
        parameterization=alpha_range,
        fixed_dataset=True, 
        batch_per_epoch=1,
        quasirandom=False,
    )
    ldc_domain.add_constraint(topWall, name="TopWall")
    #
    rightWall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        batch_size=cfg.batch_size.RightWall,
        outvar={"residual_u_comp": 0, "residual_v_comp": 0},
        # Mimicing the far field conditions "u":u_x , "v": u_y,
        criteria=~Eq(x, width / 2),  # As the right wall lies on x = width/2, we set the criteria to be x = width/2
        parameterization=alpha_range,
        fixed_dataset=True, 
        batch_per_epoch=1,
        quasirandom=False,
    )
    ldc_domain.add_constraint(rightWall, name="RightWall")
    #
    bottomWall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        batch_size=cfg.batch_size.BottomWall,
        outvar={"residual_u_comp": 0, "residual_v_comp": 0},
        # Mimicing the far field conditions "u":u_x , "v": u_y,
        criteria=~Eq(y, -height / 2),
        # As the bottom wall lies on y = -height/2, we set the criteria to be y = -height/2
        parameterization=alpha_range,
        fixed_dataset=True, 
        batch_per_epoch=1,
        quasirandom=False,
    )
    ldc_domain.add_constraint(bottomWall, name="BottomWall")

    obstacleLineAbove = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=obstacle_above,
        batch_size=cfg.batch_size.obstacle_above,
        outvar={"residual_u": u_x, 'residual_obstacle_above': 0},
        # Setting up the no slip condition for the obstacle.
        lambda_weighting={"residual_u": 100, "residual_obstacle_above": 100},  # Symbol("sdf")},
        # check Symbol("sdf") --> geo.sdf # Weights for the loss function.
        parameterization=alpha_range,
        fixed_dataset=True, 
        batch_per_epoch=1,
        quasirandom=False,
    )
    ldc_domain.add_constraint(obstacleLineAbove, name="obstacleLineAbove")

    obstacleLineBelow = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=obstacle_below,
        batch_size=cfg.batch_size.obstacle_below,
        outvar={"u": u_x, 'residual_obstacle_below': 0},
        lambda_weighting={"u": 100, "residual_obstacle_below": 100},
        parameterization=alpha_range,
        fixed_dataset=True, 
        batch_per_epoch=1,
        quasirandom=False,
    )
    ldc_domain.add_constraint(obstacleLineBelow, name="obstacleLineBelow")

    l = lambda x : x / (3 * cfg.custom.obstacle_length)  # x = 0 at the trailing edge of the obstacle

    wakeLine1_Above = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wake1_above,
        batch_size=cfg.batch_size.wake1_above,
        outvar={"u": u_x, "v": u_y * l(x)},
        lambda_weighting={"u": 100, "v": 100},
        parameterization=alpha_range,
        fixed_dataset=True, 
        batch_per_epoch=1,
        quasirandom=False,
    )
    ldc_domain.add_constraint(wakeLine1_Above, name="wakeLine1_Above")
    #
    wakeLine2_Above = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wake2_above,
        batch_size=cfg.batch_size.wake2_above,
        outvar={"u": u_x, "v": u_y * l(x)},
        lambda_weighting={"u": 100, "v": 100},
        parameterization=alpha_range,
        fixed_dataset=True, 
        batch_per_epoch=1,
        quasirandom=False,
    )
    ldc_domain.add_constraint(wakeLine2_Above, name="wakeLine2_Above")
    #
    wakeLine3_Above = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wake3_above,
        batch_size=cfg.batch_size.wake3_above,
        outvar={"u": u_x, "v": u_y * l(x)},
        lambda_weighting={"u": 100, "v": 100},
        parameterization=alpha_range,
        fixed_dataset=True, 
        batch_per_epoch=1,
        quasirandom=False,
    )

    ldc_domain.add_constraint(wakeLine3_Above, name="wakeLine3_Above")
    #
    wakeLine1_Below = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wake1_below,
        batch_size=cfg.batch_size.wake1_below,  # batch_size=150 * 2
        outvar={"u": u_x, "v": u_y * l(x)},
        lambda_weighting={"u": 100, "v": 100},
        parameterization=alpha_range,
        fixed_dataset=True, 
        batch_per_epoch=1,
        quasirandom=False,
    )
    ldc_domain.add_constraint(wakeLine1_Below, name="wakeLine1_Below")

    wakeLine2_Below = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wake2_below,
        outvar={"u": u_x, "v": u_y * l(x)},
        batch_size=cfg.batch_size.wake2_below,  # batch_size=150 * 2
        lambda_weighting={"u": 100, "v": 100},
        parameterization=alpha_range,
        fixed_dataset=True, 
        batch_per_epoch=1,
        quasirandom=False,
    )
    ldc_domain.add_constraint(wakeLine2_Below, name="wakeLine2_Below")

    wakeLine3_Below = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wake3_below,
        outvar={"u": u_x, "v": u_y * l(x)},
        batch_size=cfg.batch_size.wake1_above,  # batch_size=150 * 2
        lambda_weighting={"u": 100, "v": 100},
        parameterization=alpha_range,
        fixed_dataset=True, 
        batch_per_epoch=1,
        quasirandom=False,
    )
    ldc_domain.add_constraint(wakeLine3_Below, name="wakeLine3_Below")

    interior = PointwiseInteriorConstraint(
        geometry=geo,
        nodes=nodes,
        outvar={"Poisson_2D": 0, "residual_u": 0, "residual_v": 0},
        bounds=geo.bounds.bound_ranges,
        lambda_weighting={
            "Poisson_2D": Symbol("sdf"),
            "residual_u": Symbol("sdf"),
            "residual_v": Symbol("sdf"),
        },
        parameterization=alpha_range,
        fixed_dataset=True, 
        batch_per_epoch=1,
        quasirandom=False,
        loss=PotentialLoss(ldc_domain, cfg, arch, alpha_range),
        batch_size=cfg.batch_size.Interior
    )

    ldc_domain.add_constraint(interior, name="interior")

    return ldc_domain
