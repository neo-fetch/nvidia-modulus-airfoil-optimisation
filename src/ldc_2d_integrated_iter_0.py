# Importing relevant libraries
from sympy import Symbol, Eq, Ge, Abs, Function, Number, sin, cos
from modulus.pdes import PDES
from modulus.variables import Variables
from modulus.solver import Solver
from modulus.dataset import TrainDomain, InferenceDomain
from modulus.data import Inference
from modulus.sympy_utils.geometry_2d import Rectangle, Line
from modulus.controller import ModulusController
import numpy as np
import sys
# Importing siddharth's code
from kd_tree import kd_Tree # (X, D, N, n, point, Du, U)

# import math
# def get_angle(theta, magnitude):
#     return math.cos(theta)*magnitude, math.sin(theta)*magnitude

# ---------------------------------------------------------------------------------------------------------------------- #
# The PDES class allows you to write
# the equations symbolically in Sympy. This allows users to quickly write their equations in the most natural way possible.
# The Sympy equations are converted to TensorFlow expressions in the back-end and can also be printed to ensure correct
# implementation.

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

# ---------------------------------------------------------------------------------------------------------------------- #

# params for domain
obstacle_length = 0.10
height = 6*obstacle_length  
width = 6*obstacle_length

# We begin by defining the required variables for the geometry and then generating the 2D square geometry by instantiating
# an object of Rectangle type.
# In Modulus, a Rectangle class is defined using the coordinates for two opposite corner
# points. 
# A Line class is defined using the coordinates for two points on the line and +1 or -1 for the direction of the line. 
# The below code shows the process of generating a our required geometry in Modulus.

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
# y_range_above = {y: lambda batch_size: np.full((batch_size, 1), np.random.uniform(0, height/2.0))}
# y_range_below = {y: lambda batch_size: np.full((batch_size, 1), np.random.uniform(-height/2.0, 0))}
fixed_param_range = {alpha: lambda batch_size: np.full((batch_size, 1), np.random.uniform(-np.pi*10/180, np.pi*10/180))}


# For generating a boundary condition in Modulus, we need to sample the points on
# the required boundary/surface of the geometry and then assign them the desired values.

# A boundary can be sampled using the boundary_bc function. This will, however, sample the entire boundary of the
# geometry, in this case, all the sides of the rectangle. A particular boundary of the geometry can be sub-sampled by
# using a particular criterion for the boundary_bc using the criteria parameter. For example, to sample the top wall,
# criteria is set to y=height/2

# The desired values for the boundary condition are listed as a dictionary in outvar_sympy. In the Modulus framework,
# we define these variables as keys of this dictionary which are converted to appropriate nodes in the computational graph.
# The number of points to sample on each boundary are specified using the batch_size_per_area parameter.

# Weights to any variables can be specified as an input to the lambda_sympy parameter. A keyword ’lambda_’ is added
# onto the key names specified in the outvar_sympy to specify the corresponding weights in the total loss function.
# For example, if we want to weight the loss function by the magnitude of the velocity, we can specify the weights as
# {'lambda_u': 1, 'lambda_v': 1, 'lambda_phi': 1}

# Equations to solve:
# The conservation equations of fluid mechanics are enforced on all the points in the
# interior of the geometry. 

# Similar to sampling boundaries, we will use interior_bc function to sample points in the interior of a geometry. The
# equations to solve are specified as a dictionary input to outvar_sympy.

# The parameter bounds, determines the range for sampling the values for variables x and y. The lambda_sympy
# parameter is used to determine the weights for different losses. In this problem, we weight each equation at each point
# by its distance from the boundary by using the Signed Distance Field (SDF) of the geometry. This implies that the
# points away from the boundary are weighted higher compared to the ones closer to the boundary. We found that this
# type of weighting of the loss functions leads to a faster convergence since it avoids discontinuities at the boundaries

# ---------------------------------------------------------------------------------------------------------------------- #

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
        # flow_rate = u_x*width + u_y*height

        # Left wall
        leftWall = geo.boundary_bc(
            outvar_sympy={"residual_u_comp": 0, "residual_v_comp": 0}, # "u": u_x, "v": u_y, 
            batch_size_per_area=250*2,
            criteria=Eq(x, -width / 2),
            param_ranges ={**fixed_param_range},
            fixed_var=False
        )
        self.add(leftWall, name="LeftWall")

        # outlet
        # outletBC = geo.boundary_bc(
        #     outvar_sympy={"residual_u_comp": 0, "residual_v_comp": 0}, # Mimicing the far field conditions "u":u_x , "v": u_y, 
        #     batch_size_per_area=500*2,
        #     criteria=Ge(y/height+x/width, 1/2),
        #     param_ranges ={**fixed_param_range},
        #     fixed_var=False
        # )
        # self.add(outletBC, name="Outlet")

        # Top wall
        topWall = geo.boundary_bc(
            outvar_sympy={"residual_u_comp": 0, "residual_v_comp": 0}, # "u": u_x, "v": u_y,
            batch_size_per_area=250*2,
            criteria=Eq(y, height / 2),
            param_ranges ={**fixed_param_range},
            fixed_var=False
        )
        self.add(topWall, name="TopWall")

        # Right wall
        rightWall = geo.boundary_bc(
            outvar_sympy={"residual_u_comp": 0, "residual_v_comp": 0}, # "u": u_x, "v": u_y,
            batch_size_per_area=250*2,
            criteria=Eq(x, width / 2),
            param_ranges ={**fixed_param_range},
            fixed_var=False
        )
        self.add(rightWall, name="RightWall")

        # Bottom Wall
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
            lambda_sympy={"lambda_u": 100, "lambda_residual_obstacle_above": geo.sdf},
            param_ranges ={**fixed_param_range},
            # param_ranges ={**y_range_above, **fixed_param_range},
            fixed_var=False            
        )
        self.add(obstacleLineAbove, name="obstacleLineAbove")

        # obstacleLine Below
        obstacleLineBelow = obstacle_below.boundary_bc(
            outvar_sympy={"u": u_x, 'residual_obstacle_below': 0},
            batch_size_per_area=600*2,
            lambda_sympy={"lambda_u": 100, "lambda_residual_obstacle_below": geo.sdf},
            param_ranges ={**fixed_param_range},
            # param_ranges ={**y_range_below, **fixed_param_range},
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
            param_ranges ={**fixed_param_range},
            # param_ranges ={**y_range_above, **fixed_param_range},
            fixed_var=False            
        )
        self.add(wakeLine1_Above, name="wakeLine1_Above")

        wakeLine2_Above = wake2_above.boundary_bc(
            outvar_sympy={"u": u_x, "v": u_y*l(x), 'residual_obstacle_wake2_above': 0},
            batch_size_per_area=150*2,
            lambda_sympy={"lambda_u": 100, "lambda_v": 100, "lambda_residual_obstacle_wake2_above": geo.sdf},
            param_ranges ={**fixed_param_range},
            # param_ranges ={**y_range_above, **fixed_param_range},
            fixed_var=False
        )
        self.add(wakeLine2_Above, name="wakeLine2_Above")

        wakeLine3_Above = wake3_above.boundary_bc(
            outvar_sympy={"u": u_x, "v": u_y*l(x), 'residual_obstacle_wake3_above': 0},
            batch_size_per_area=150*2,
            lambda_sympy={"lambda_u": 100, "lambda_v": 100, "lambda_residual_obstacle_wake3_above": geo.sdf},
            param_ranges ={**fixed_param_range},
            # param_ranges ={**y_range_above, **fixed_param_range},
            fixed_var=False
        )

        self.add(wakeLine3_Above, name="wakeLine3_Above")

        wakeLine1_Below = wake1_below.boundary_bc(
            outvar_sympy={"u": u_x, "v": u_y*l(x), 'residual_obstacle_wake1_below': 0},
            batch_size_per_area=150*2,
            lambda_sympy={"lambda_u": 100, "lambda_v": 100, "lambda_residual_obstacle_wake1_below": geo.sdf},
            param_ranges ={**fixed_param_range},
            # param_ranges ={**y_range_below, **fixed_param_range},
            fixed_var=False
        )
        self.add(wakeLine1_Below, name="wakeLine1_Below")

        wakeLine2_Below = wake2_below.boundary_bc(
            outvar_sympy={"u": u_x, "v": u_y*l(x), 'residual_obstacle_wake2_below': 0},
            batch_size_per_area=150*2,
            lambda_sympy={"lambda_u": 100, "lambda_v": 100, "lambda_residual_obstacle_wake2_below": geo.sdf},
            param_ranges ={**fixed_param_range},
            # param_ranges ={**y_range_below, **fixed_param_range},
            fixed_var=False
        )
        self.add(wakeLine2_Below, name="wakeLine2_Below")

        wakeLine3_Below = wake3_below.boundary_bc(
            outvar_sympy={"u": u_x, "v": u_y*l(x), 'residual_obstacle_wake3_below': 0},
            batch_size_per_area=150*2,
            lambda_sympy={"lambda_u": 100, "lambda_v": 100, "lambda_residual_obstacle_wake3_below": geo.sdf},
            param_ranges ={**fixed_param_range},
            # param_ranges ={**y_range_below, **fixed_param_range},
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
            batch_size_per_area=4000*2,
            param_ranges ={**fixed_param_range},
            fixed_var=False            
        )
        self.add(interior, name="interior")

        # neighborhood = geo.interior_bc(
        #     outvar_sympy={"Poisson_2D": 0, "residual_u": 0, "residual_v": 0},
        #     bounds={x: (-height / 3, height / 3), y: (-height / 8, height / 8)},
        #     lambda_sympy={
        #         "lambda_Poisson_2D": geo.sdf,
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
        x, y, alpha = Symbol('x'), Symbol('y'), Symbol('alpha')
        interior = Inference(geo.sample_interior(10000, bounds={x: (-width / 2, width / 2), y: (-height / 2, height / 2)}, param_ranges={alpha: np.pi*(10/180)}), ['u', 'v', 'phi'])
        self.add(interior, name="Inference")

# ---------------------------------------------------------------------------------------------------------------------- #

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

    # The following function allows you to interpolate your vector points using
    # the interpolation function as weighted average of the vector points using the weight array
    # which is a function of inverse square of the distance from the neighbor point to the interpolation point.
    # Credits: Siddarth Agarwal
    def phi_interpolation(phi, n, weigth_arr):
        interpolated_phi_x = 0
        phi_x_numer = 0
        phi_x_denom = 0
        flag_val = 0 #To check whether the interpolation function follows the condition when distance == 0.
        for i in range(0,n):
            if(dist[i]==0):
                interpolated_phi_x = phi[i]
                flag_val = 1
                break
            else:
                phi_x_numer = phi_x_numer + (weigth_arr[i]*phi[i])
                phi_x_denom = phi_x_denom + weigth_arr[i]
        if(flag_val==1):
            return(interpolated_phi_x)
        elif(flag_val==0 and phi_x_denom!=0):
            return(phi_x_numer/phi_x_denom)

    # The following function generates a sub point cloud using the given point cloud 
    # and the given bounding box around the given point coordinate.
    def get_sub_pc(self, point, band, x_range, y_range):
        x_range = [[point[0] - x_range*obstacle_length, -obstacle_length][point[0] - x_range*obstacle_length < -obstacle_length], \
            [point[0] + x_range*obstacle_length, width/2][point[0] + x_range*obstacle_length > width/2]]

        y_range = sorted([0, point[1] + y_range*obstacle_length])
        
        sub_pc = []
        for i in range(len(band)):
            if x_range[0] <= band[i][0] <= x_range[1] and y_range[0] <= band[i][1] <= y_range[1]:
                sub_pc.append(band[i])
        return sub_pc

    def custom_loss(self, domain_invar, pred_domain_outvar, true_domain_outvar, step):
        x_interior = domain_invar['interior']['x'] + domain_invar['RightWall']['x'] # x coordinate of interior points
        y_interior = domain_invar['interior']['y'] + domain_invar['RightWall']['y'] # y coordinate of interior points

        x_wkeobs_above = domain_invar['obstacleLineAbove']['x'] + domain_invar['wakeLine1_Above']['x'] +\ 
             domain_invar['wakeLine2_Above']['x'] + domain_invar['wakeLine3_Above']['x'] # x coordinate of obstacle and wake points above
        y_wkeobs_above = domain_invar['obstacleLineAbove']['y'] + domain_invar['wakeLine1_Above']['y'] +\
             domain_invar['wakeLine2_Above']['y'] + domain_invar['wakeLine3_Above']['y'] # y coordinate of obstacle and wake points above

        x_wkeobs_below = domain_invar['obstacleLineBelow']['x'] + domain_invar['wakeLine1_Below']['x'] +\
             domain_invar['wakeLine2_Below']['x'] + domain_invar['wakeLine3_Below']['x'] # x coordinate of obstacle and wake points below
        y_wkeobs_below = domain_invar['obstacleLineBelow']['y'] + domain_invar['wakeLine1_Below']['y'] +\
             domain_invar['wakeLine2_Below']['y'] + domain_invar['wakeLine3_Below']['y'] # y coordinate of obstacle and wake points below

        # wkeobs_above = np.asarray([x_wkeobs_above, y_wkeobs_above]).T
        # wkeobs_below = np.asarray([x_wkeobs_below, y_wkeobs_below]).T
        
        # zip() by default stores a list of tuples. We need to convert the tuples into lists.
        wkeobs_above = list(zip(x_wkeobs_above, y_wkeobs_above))
        wkeobs_above = [list(t) for t in wkeobs_above] # convert from tuple to list
        wkeobs_below = list(zip(x_wkeobs_below, y_wkeobs_below))
        wkeobs_below = [list(t) for t in wkeobs_below] # convert from tuple to list
        interior = list(zip(x_interior, y_interior))
        interior = [list(t) for t in interior] # convert from tuple to list

        ################################################################################################################

        # EXPLANATION FOR CODE ABOVE:
        # To form the band we need around the obstacle and wake lines, we need to know 
        # the x and y values of all the interior points, obstacle lines, wake lines and right wall in the event of cut off. 
        # The code above does this by adding the x and y values of all the relevant regions.

        # We now move on to filtering our points to match our criteria for selection of points around the obstacle and wake lines. 
        # A diagram below shows us the band of points around the obstacle and wake lines.

        # +-------------------+
        # |                   |
        # |                   |
        # |     +-------------+
        # |     |/////////////|
        # |     |---==========|
        # |     |\\\\\\\\\\\\\|
        # |     +-------------+
        # |                   |
        # |                   |
        # +-------------------+
        
        # Diagram: Band around obstacle and wake lines

        # We need to filter the interior points to only select those that lie within the square. 
        # We do this by taking the x and y values of the interior points
        # and comparing them to the x and y values of the square. 
        # If the x and y values are within the square, we add them to the interior points.

        ################################################################################################################

        band = [] # This is the band of points around the obstacle and wake lines.
        band_range_x = [-obstacle_length, width/2] # This is the range of x values of the band.
        band_range_y = [-0.15, 0.15] # This is the range of y values of the band.

        # The code below filters the interior points to only select those that lie within the range of the band.
        for i in range(len(x_interior)):
            if band_range_x[0] <= x_interior[i] <= band_range_x[1] and band_range_y[0] <= y_interior[i] <= band_range_y[1]:
                band.append([x_interior[i], y_interior[i]])

        # We now have all the points within the band. We need to divide the band into above and below y = 0. 
        # We do this by creating two lists, one for above y = 0 and one for below y = 0.

        band_above = [] # This is the band of points above y = 0.
        band_below = [] # This is the band of points below y = 0.

        for i in range(len(band)):
            if band[i][1] > 0:
                band_above.append(band[i]) # If the y value of the point is above y = 0, we add it to the band above.
            else if band[i][1] < 0:
                band_below.append(band[i]) # If the y value of the point is below y = 0, we add it to the band below.
        
        # We dont concern ourselves with the points that are exactly on the y = 0 line as we are going to add obstacle and wake points 
        # separately in the code below.

        band_above = band_above + wkeobs_above # We add the obstacle and wake points to the band above.
        band_below = band_below + wkeobs_below # We add the obstacle and wake points to the band below.
        
        # Now that we have divided the band into above and below y = 0, we can now start our process of dividing 
        # the point cloud into smaller sub point clouds using my good friend sid's subroutine.

        band_total = band_above + band_below # We combine the band above and below into one list.
        bands = [band_above, band_below] # We create a list of the above and below band to iterate through them later.
        dx = 0.015*obstacle_length # This is the distance between the main point and the constructed points across the x axis.
        dy = 0.015*obstacle_length # This is the distance between the main point and the constructed point across the y axis.
        weights = [] # This is the list of weights of the neighbors used to interpolate the phi value of the constructed points.
        neighbors = [] # These are the neighbors of the constructed points n.
        for i in range(2):
            for j in range(len(bands[i])):
                xfy = [bands[i][j][0] + dx, bands[i][j][1]] # This is the x in front of the original point.
                xby = [bands[i][j][0] - dx, bands[i][j][1]] # This is the x behind the original point.
                xyf = [bands[i][j][0], bands[i][j][1] + (-1)**i*dy] # This is the y next of the original point.
                xy = [bands[i][j][0], bands[i][j][1]] # This is the original point.

                Nxfy = get_sub_pc(xfy, bands[i], 0.3, (-1)**i*0.6) # This is the sub point cloud of the x in front of the original point.
                Nxby = get_sub_pc(xby, bands[i], 0.3, (-1)**i*0.6) # This is the sub point cloud of the x behind the original point.
                Nxy = get_sub_pc(xy, bands[i], 0.3, (-1)**i*0.6) # This is the sub point cloud of the original point.
                Nxyf = Nxy # This is the sub point cloud of the y next of the original point, and is the same as Nxy.

                # We now use sid's subroutine to get the neighbors and weights of the constructed points. 
                # Sid's subroutine uses KD tree to find the neighbors and their corresponding weights for the constructed points.
                # The weights are calculated using the inverse-square of the distance between the constructed point and the neighbors.
                # The neighbors are for now taken as 7 points.
                # The neighbors may be points that are within the band, or outside of it.

                Wxfy, neigh_xfy = kd_Tree(Nxfy, 2, len(Nxfy), 7, xfy)
                Wxby, neigh_xby = kd_Tree(Nxby, 2, len(Nxby), 7, xby)
                Wxyf, neigh_xyf = kd_Tree(Nxyf, 2, len(Nxyf), 7, xyf)
                # Wxy, neigh_xy = kd_Tree(Nxy, 2, len(Nxy), 7, xy)

                weights.append([Wxfy, Wxby, Wxyf]) # We add the weights of the neighbors to the weights list.
                neighbors.append([neigh_xfy, neigh_xby, neigh_xyf, xy]) # We add the neighbors to the neighbors list.
        
        # In the code above, we need 4 points: two points on the x-axis for central differentiation and one 
        # point on the y-axis for backward differentiation. We then use these points to find the sub point cloud around them. 
        # Using these sub point clouds we can then calculate the weights and neighbors for each point. 
        # The weights and neighbors are calculated using the kdTree function(sid's subroutine). 
        # We then store them in the weights and neighbors list, which will be used later. 
        # For now each entry in weight and neighbor corresponds to information about 4 points: xfy, xby, xyf, xy. 
        # Since we specified the neighbors as 7, we will have 7 dimensional vector for each of theses points, 
        # making it a total of 4 times 7 = 28 entries per point (x,y).

        u_band = [] # This is the list that stores the u values within the band initially. 
        v_band = [] # This is the list that stores the v values within the band initially.
        for i in range(len(neighbors[0])):
            # xfy
            x_xfy = [x for x,y in neighbors[i][0]] # We create a list of the x values of the neighbors of xfy.
            y_xfy = [y for x,y in neighbors[i][0]] # We create a list of the y values of the neighbors of xfy.

            # xby
            x_xby = [x for x,y in neighbors[i][1]] # We create a list of the x values of the neighbors of xby.
            y_xby = [y for x,y in neighbors[i][1]] # We create a list of the y values of the neighbors of xby.

            # xyf
            x_xyf = [x for x,y in neighbors[i][2]] # We create a list of the x values of the neighbors of xyf.
            y_xyf = [y for x,y in neighbors[i][2]] # We create a list of the y values of the neighbors of xyf.

            # phi xfy
            phi_xfy = self.nets[0].evaluate({'x': x_xfy, 'y': y_xfy})['phi'] # We evaluate the phi values of the neighbors of xfy using the neural network.

            # phi xby
            phi_xby = self.nets[0].evaluate({'x': x_xby, 'y': y_xby})['phi'] # We evaluate the phi values of the neighbors of xby using the neural network.

            # phi xyf
            phi_xyf = self.nets[0].evaluate({'x': x_xyf, 'y': y_xyf})['phi'] # We evaluate the phi values of the neighbors of xyf using the neural network.

            # phi xy
            phi_xy = self.nets[0].evaluate({'x': xy[0], 'y': xy[1]})['phi'] # We evaluate the phi values of the original point using the neural network.

            # Interpolating phi for xfy, xby, xyf using phi_interpolation()

            phi_xfy = phi_interpolation(phi_xfy, len(phi_xfy), weights[i][0]) # We interpolate the phi values of the neighbors of xfy using the weighted average.

            phi_xby = phi_interpolation(phi_xby, len(phi_xby), weights[i][1]) # We interpolate the phi values of the neighbors of xby using the weighted average.

            phi_xyf = phi_interpolation(phi_xyf, len(phi_xyf), weights[i][2]) # We interpolate the phi values of the neighbors of xyf using the weighted average.

            # Calculating u and v
            u_band.append([xy[0], xy[1]], [(phi_xfy - phi_xby)/(2*dx)]) # We calculate the u value of the original point using the central difference formula.
            v_band.append([xy[0], xy[1]], [abs(phi_xyf - phi_xy)/(dy)]) # We calculate the v value of the original point using the forward/backward difference formula.

        # We now have the u and v values for each point in the point cloud. We now need to find the u and v values outside the band.
        # We do this by finding the points that are outside the band and then using the neural network to evaluate them.

        # We first find the points that are outside the band.

        x_outside = [x for x,y in interior if x,y not in band_total]
        y_outside = [y for x,y in interior if x,y not in band_total]
                
        # We now use the neural network(self.nets[0]) to evaluate the points that are outside the band.
        phi_outside = self.nets[0].evaluate({'x': x_outside, 'y': y_outside})['phi']

        u_interior = tf.gradients(phi_outside, x_outside)[0] # We calculate the u values of the points outside the band using d(phi)/dx (automatic differentiation).
        v_interior = tf.gradients(phi_outside, y_outside)[0] # We calculate the v values of the points outside the band using d(phi)/dy (automatic differentiation).

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

# ---------------------------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    ctr = ModulusController(PotentialSolver)
    ctr.run()