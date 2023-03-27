import os
from pathos.pools import ThreadPool as TPool
import torch
from modulus.hydra import ModulusConfig
from sklearn.neighbors import KDTree
import numpy as np
from modulus.loss import Loss
# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def kd_tree(X, n, point):
    # The 2 lines below uses the python inbuilt library for KD-Tree.
    tree = KDTree(X)
    dist, ind = tree.query([point], k=n)
    # dist: is a 2-D vector which stores the distance value of all the 'n' neighbouring points from x.
    # ind: is a 2-D vector which stores the indices of all the 'n' neighbouring points from x.

    neighbour_points = []
    for i in range(0, n):
        neighbour_points.append(X[ind[0][i]])

    # Since all the weight values will be fixed throughout the code,
    # therefore storing all the weights of n-nearest neighbours in array.
    weigth_arr = np.zeros(n, dtype=float)
    for i in range(0, n):
        if dist[0][i] != 0:
            weigth_arr[i] = 1 / (dist[0][i] ** 2)
    return weigth_arr, neighbour_points, dist


@torch.no_grad()
def get_sub_pc(point, band, x_range, y_range, cfg):
    width = cfg.custom.unscaled_domain_height * cfg.custom.obstacle_length
    # Set the x range of the sub point cloud
    x_range = [[point[0] - x_range * cfg.custom.obstacle_length, -cfg.custom.obstacle_length]
               [point[0] - x_range * cfg.custom.obstacle_length < -cfg.custom.obstacle_length],
               [point[0] + x_range * cfg.custom.obstacle_length, width / 2]
               [point[0] + x_range * cfg.custom.obstacle_length > width / 2]]

    # Set the y range of the sub point cloud
    y_range = sorted([0, point[1] + y_range * cfg.custom.obstacle_length])

    i = (x_range[0] <= band[:, 0]) & (band[:, 0] <= x_range[1])
    j = (y_range[0] <= band[:, 1]) & (band[:, 1] <= y_range[1])
    sub_pc = band[i & j]
    return sub_pc


@torch.no_grad()
def phi_interpolation(phi, n, weigth_arr, dist):
    interpolated_phi_x = 0  # Initialize the interpolated value of phi_x
    phi_x_numer = 0  # Initialize the numerator of the interpolated value of phi_x
    phi_x_denom = 0  # Initialize the denominator of the interpolated value of phi_x
    flag_val = 0  # To check whether the interpolation function follows the condition when distance == 0.

    for i in range(0, n):
        if dist[0][i] == 0:
            interpolated_phi_x = phi[i]  # If the distance is zero, then the interpolated value of phi_x
            flag_val = 1  # is the value of the point itself.
            break
        else:
            phi_x_numer = phi_x_numer + (weigth_arr[i] * phi[i])  # If the distance is not zero,
            phi_x_denom = phi_x_denom + weigth_arr[i]  # then the numerator and denominator of the
    if flag_val == 1:
        # interpolated value of phi_x are calculated using weighted average function.
        return interpolated_phi_x
    elif flag_val == 0 and phi_x_denom != 0:
        # If the interpolation function does not follow the condition when distance == 0
        return phi_x_numer / phi_x_denom


@torch.no_grad()
def pull_coordinates(domain_invar):
    """
    This function pulls the coordinates from the geometry curves.
    Parameters
    ----------
    domain_invar: it contains overall geometry information.

    Returns
    -------
    total domain: x and y coordinates of the total domain.
    """
    total_domain = {}
    for key in domain_invar.constraints.keys():
        total_domain[key] = {}
        # temp = {'x': domain_invar.constraints[key].dataset.invar_fn()['x'],
        #         'y': domain_invar.constraints[key].dataset.invar_fn()['y']}
        temp = {'x': domain_invar.constraints[key].dataset.invar['x'],
                'y': domain_invar.constraints[key].dataset.invar['y']}
        total_domain[key].update(temp)

    return total_domain


@torch.no_grad()
def prepare_coordinates(total_domain, interior_invar):
    ################################################################################################################

    # EXPLANATION FOR CODE ABOVE:
    # To form the band around the obstacle and wake lines, we need to know
    # the x and y values of all the interior points, obstacle lines, wake lines and right wall in the event of cut off.
    # The code above does this by adding the x and y values of all the relevant regions.

    # We now move on to filtering our points to match our criteria for selection of points around the obstacle and
    # wake lines. A diagram below shows us the band of points around the obstacle and wake lines.

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
    ################################################################################################################

    x_interior = torch.concat([interior_invar['x'].cpu(), total_domain['RightWall']['x']], dim=0).detach()
    # Concatenate the interior and the right wall for x. interior_invar variable for interior is taken on purpose
    # instead of total_domain['interior']. The total_domain['interior'] are shuffled and tedious to arrange them again.
    y_interior = torch.concat([interior_invar['y'].cpu(), total_domain['RightWall']['y']], dim=0).detach()
    # Concatenate the interior and the right wall for y

    x_wkeobs_above = torch.concat([total_domain['obstacleLineAbove']['x'], total_domain['wakeLine1_Above']['x'],
                                   total_domain['wakeLine2_Above']['x'], total_domain['wakeLine3_Above']['x']],
                                  dim=0)  # x coordinate of obstacle and wake points above
    y_wkeobs_above = torch.concat([total_domain['obstacleLineAbove']['y'], total_domain['wakeLine1_Above']['y'],
                                   total_domain['wakeLine2_Above']['y'], total_domain['wakeLine3_Above']['y']],
                                  dim=0)  # y coordinate of obstacle and wake points above

    x_wkeobs_below = torch.concat([total_domain['obstacleLineBelow']['x'], total_domain['wakeLine1_Below']['x'],
                                   total_domain['wakeLine2_Below']['x'], total_domain['wakeLine3_Below']['x']],
                                  dim=0)  # x coordinate of obstacle and wake points below
    y_wkeobs_below = torch.concat([total_domain['obstacleLineBelow']['y'], total_domain['wakeLine1_Below']['y'],
                                   total_domain['wakeLine2_Below']['y'], total_domain['wakeLine3_Below']['y']],
                                  dim=0)  # y coordinate of obstacle and wake points below

    wkeobs_above = torch.hstack([x_wkeobs_above, y_wkeobs_above])
    wkeobs_below = torch.hstack([x_wkeobs_below, y_wkeobs_below])

    interior = torch.hstack([x_interior, y_interior])

    return interior, wkeobs_above, wkeobs_below


@torch.no_grad()
def init_domain(interior_invar, domain_invar, cfg=ModulusConfig):
    total_domain = pull_coordinates(domain_invar)

    interior, wkeobs_above, wkeobs_below = prepare_coordinates(total_domain, interior_invar)

    del total_domain, interior_invar

    width = cfg.custom.unscaled_domain_height * cfg.custom.obstacle_length
    band_range_x = [-cfg.custom.obstacle_length, width / 2]
    # This is the range of x values of the band.
    band_range_y_belt = [cfg.custom.band_range_y_belt[0], cfg.custom.band_range_y_belt[1]]
    # This is the range of y values of the belt.
    band_range_y = [cfg.custom.band_range_y[0], cfg.custom.band_range_y[1]]
    # This is the range of y values of the band.

    # The code below filters the interior points to only select those that lie within the range of the band.
    # We do this by taking the x and y values of the interior points
    # and comparing them to the x and y values of the band.
    # If the x and y values are within the band, assign them to the band.

    ii = (band_range_x[0] <= interior[:, 0]) & (interior[:, 0] <= band_range_x[1])
    jj = (band_range_y[0] <= interior[:, 1]) & (interior[:, 1] <= band_range_y[1])
    band = interior[ii & jj]

    # The code below filters the interior points to only select those that lie within the range of the belt.
    ii = (band_range_x[0] <= band[:, 0]) & (band[:, 0] <= band_range_x[1])
    jj = (band_range_y_belt[0] <= band[:, 1]) & (band[:, 1] <= band_range_y_belt[1])
    belt = band[ii & jj]

    # While band is to get the sub point cloud, the points within belt is used for finite difference calculation.
    # We now have all the points within the band. We need to divide the band into above and below y = 0.

    ii = band[:, 1] > 0  # If the y value of the point is above y = 0, assign it to the band above.
    band_above = band[ii]
    jj = band[:, 1] < 0  # If the y value of the point is below y = 0, assign it to the band below.
    band_below = band[jj]

    # We don't concern ourselves with the points that are exactly on the y = 0 line as we are going to add obstacle
    # and wake points separately in the code below.
    ii = belt[:, 1] > 0
    belt_above = belt[ii]
    jj = belt[:, 1] < 0
    belt_below = belt[jj]

    band_above = torch.concat([band_above, wkeobs_above], dim=0)
    band_below = torch.concat([band_below, wkeobs_below], dim=0)

    belt_above = torch.concat([belt_above, wkeobs_above], dim=0)
    belt_below = torch.concat([belt_below, wkeobs_below], dim=0)

    # Now that we have divided the band into above and below y = 0, we can now start our process of dividing
    # the point cloud into smaller sub point clouds using my good friend sid's subroutine.

    belt_total = torch.concat([belt_above, belt_below], dim=0)
    bands = [band_above, band_below]  # list of the above and below band to iterate through them later.
    belts = [belt_above, belt_below]  # list of the above and below belt to iterate through them later.
    dx = cfg.custom.dx  # distance between the main point and the constructed points across the x-axis.
    dy = cfg.custom.dy  # distance between the main point and the constructed point across the y-axis.
    weights = []  # list of weights of the neighbors used to interpolate the phi value of the constructed points.
    neighbors = []  # These are the neighbors of the constructed points n.
    distance = []

    del belt_above, belt_below, band_above, band_below, ii, jj

    # Following loop manages the above and below vector of belts independently. So, it's hardcoded.
    for i in range(2):
        @torch.no_grad()
        def neigh_weigh_dist(j):
            xy = [belts[i][j][0], belts[i][j][1]]  # This is the original point.
            xfy = [xy[0] + dx, xy[1]]  # x in front of the original point.
            xby = [xy[0] - dx, xy[1]]  # x behind the original point.
            xyf = [xy[0], xy[1] + (-1) ** i * dy]  # y next of the original point.
            xyff = [xy[0], xy[1] + (-1) ** i * (2 * dy)]  # y next to next of the original point.

            Nxfy = get_sub_pc(xfy, bands[i], cfg.custom.x_range_sub_pc, (-1) ** i * cfg.custom.y_range_sub_pc, cfg)
            # sub point cloud of the x in front of the original point.
            Nxby = get_sub_pc(xby, bands[i], cfg.custom.x_range_sub_pc, (-1) ** i * cfg.custom.y_range_sub_pc, cfg)
            # sub point cloud of the x behind the original point.
            Nxy = get_sub_pc(xy, bands[i], cfg.custom.x_range_sub_pc, (-1) ** i * cfg.custom.y_range_sub_pc, cfg)
            # sub point cloud of the original point.
            Nxyf = Nxyff = Nxy  # sub point cloud of the y next of the original point, and is the same as Nxy.

            # The weights are calculated using the inverse-square of the distance between the constructed point and
            # the neighbors. The neighbors may be points that are within the band, or outside it.
            Wxfy, neigh_xfy, dist_xfy = kd_tree(X=Nxfy, n=cfg.custom.neigh_point_kd_tree, point=xfy)
            Wxby, neigh_xby, dist_xby = kd_tree(Nxby, cfg.custom.neigh_point_kd_tree, xby)
            Wxyf, neigh_xyf, dist_xyf = kd_tree(Nxyf, cfg.custom.neigh_point_kd_tree, xyf)
            Wxyff, neigh_xyff, dist_xyff = kd_tree(Nxyff, cfg.custom.neigh_point_kd_tree, xyff)

            # In the code above, we need 4 points: two points on the x-axis for central differentiation and one point
            # on the y-axis for backward differentiation. We then use these points to find the sub point cloud around
            # them. Using these sub point clouds we can then calculate the weights and neighbors for each point. The
            # weights and neighbors are calculated using the kdTree function. We then store them in
            # the weights and neighbors list, which will be used later. For now each entry in weight and neighbor
            # corresponds to information about 4 points: xfy, xby, xyf, xy. Since we specified the neighbors as 7,
            # we will have 7 dimensional vector for each of these points, making it a total of 4 times 7 = 28
            # entries per point (x,y).

            weigths = [Wxfy, Wxby, Wxyf, Wxyff]  # weights of the neighbors added to the weights list.
            neighbours = [neigh_xfy, neigh_xby, neigh_xyf, neigh_xyff, xy]  # neighbors added to the neighbors list.
            distnace = [dist_xfy, dist_xby, dist_xyf, dist_xyff]  # distances added to the distance list.
            return weigths, neighbours, distnace

        hpool = TPool()
        hpool.nthreads = int(os.popen('nproc').read())
        results = hpool.map(neigh_weigh_dist, range(len(belts[i])))

        weights.extend([results[i][0] for i in range(len(results))])
        neighbors.extend([results[i][1] for i in range(len(results))])
        distance.extend([results[i][2] for i in range(len(results))])
    del bands, belts, results, hpool

    return weights, neighbors, distance, belt_total


@torch.no_grad()
def pull_sort_wrt_x(tensor_variable, ind):
    # since belt_interior_points are jumbled, it's not possible to multiply the corr lambda values to belt points.
    # So there is a need to aligned points with some axis. let's sort belt_interior_pts wrt x-axis in increasing
    # order and take sorted index and align all the belt residual values.

    temp = tensor_variable[ind]
    # Sorting about x-axis of belt_interior_points
    _, indx = torch.sort(temp[:, 0])
    return indx


@torch.no_grad()
def get_index(master, child):
    # https://stackoverflow.com/questions/62588779/pytorch-differences-between-two-tensors
    return master.unsqueeze(1).eq(child).all(-1).any(-1)


class PotentialLoss(Loss):
    """
    This class uses 'Loss' base class to calculate the loss of interior points as defined.
    The loss of the points within the band is calculated using the finite difference approach. However, the loss of the
    points outside the band is calculated using traditional method (Automatic differentiation).

    The primary task is to segregate the points lying within and outside the band. Then, they are treated separately.
    Finally, loss from are added and assigned to the loss dictionary and returned.

    params:
    ---------
    Input: geometry, domain interior points,
    ---------
    Returns: loss (within the band and outside the band).
    """

    def __init__(self, domain, cfg, arch, alpha_same):
        super().__init__()

        # device agnostic code
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.belt_total = None
        self.distance = None
        self.neighbors = None
        self.weights = None
        self.init_flag = 0
        self.domain = domain
        self.cfg = cfg
        self.nets = arch
        self.alpha_same = alpha_same   # check this position

    @torch.no_grad()
    def phi_evaluation_neighbor(self, x_values, y_values, xy):
        # Random value for alpha is generated
        # temp_angle = np.pi * self.cfg.custom.AoA / 180
        #
        # alpha = torch.full((self.cfg.custom.neigh_point_kd_tree, 1),
        #                    np.random.uniform(- temp_angle, temp_angle)).type(torch.float32).to(self.device)
        # alpha = torch.full((self.cfg.custom.neigh_point_kd_tree, 1), self.domain.constraints['interior'].dataset.invar['alpha'][0]).type(torch.float32).to(self.device)
        alpha = torch.tensor(np.full((self.cfg.custom.neigh_point_kd_tree, 1), self.alpha_same), dtype=torch.float32).to(self.device)

        phi_xfy = self.nets.evaluate(
            {'x': x_values[:, [0]], 'y': y_values[:, [0]], 'alpha': alpha})[
            'phi']  # We evaluate the phi values of the neighbors of xfy using the neural network.

        phi_xby = self.nets.evaluate(
            {'x': x_values[:, [1]], 'y': y_values[:, [1]], 'alpha': alpha})[
            'phi']  # We evaluate the phi values of the neighbors of xby using the neural network.
        phi_xyf = self.nets.evaluate(
            {'x': x_values[:, [2]], 'y': y_values[:, [2]], 'alpha': alpha})[
            'phi']  # We evaluate the phi values of the neighbors of xyf using the neural network.
        phi_xyff = self.nets.evaluate(
            {'x': x_values[:, [3]], 'y': y_values[:, [3]], 'alpha': alpha})[
            'phi']  # We evaluate the phi values of the neighbors of xyff using the neural network
        xy_pred = self.nets.evaluate(
            {'x': xy[0], 'y': xy[1], 'alpha': alpha[0]})

        return phi_xfy, phi_xby, phi_xyf, phi_xyff, xy_pred

    def in_belt_residuals(self):
        @torch.no_grad()
        def residuals_calc(i):
            # xfy
            x_xfy = torch.stack(self.neighbors[i][0])[:, [0]].type(torch.float32)

            # In [:,[0]], [0] is used to obtain proper shape.
            y_xfy = torch.stack(self.neighbors[i][0])[:, [1]].type(torch.float32)

            # xby
            x_xby = torch.stack(self.neighbors[i][1])[:, [0]].type(torch.float32)
            y_xby = torch.stack(self.neighbors[i][1])[:, [1]].type(torch.float32)

            # xyf
            x_xyf = torch.stack(self.neighbors[i][2])[:, [0]].type(torch.float32)
            y_xyf = torch.stack(self.neighbors[i][2])[:, [1]].type(torch.float32)

            # xyff
            x_xyff = torch.stack(self.neighbors[i][3])[:, [0]].type(torch.float32)
            y_xyff = torch.stack(self.neighbors[i][3])[:, [1]].type(torch.float32)

            # xy
            xy = torch.reshape(torch.hstack([self.neighbors[i][4][0], self.neighbors[i][4][1]]), (2, 1)).type(
                torch.float32)  # The original point.

            # --------------- Phi value evaluation ---------------- #
            phi_xfy, phi_xby, phi_xyf, phi_xyff, xy_pred = self.phi_evaluation_neighbor(
                torch.hstack([x_xfy, x_xby, x_xyf, x_xyff]).to(self.device),
                torch.hstack([y_xfy, y_xby, y_xyf, y_xyff]).to(self.device),
                xy.to(self.device)
            )

            # Assigning prediction values to corr variables
            u_pred, v_pred, phi_xy = xy_pred['u'], xy_pred['v'], xy_pred['phi']

            del x_xfy, y_xfy, x_xby, y_xby, x_xyf, y_xyf, x_xyff, y_xyff, xy_pred

            # --------------- Phi value interpolation -------------- #
            phi_xfy = phi_interpolation(phi_xfy, phi_xfy.shape[0], self.weights[i][0], self.distance[i][
                0])  # We interpolate the phi values of the neighbors of xfy using the weighted average.
            phi_xby = phi_interpolation(phi_xby, phi_xby.shape[0], self.weights[i][1], self.distance[i][
                1])  # We interpolate the phi values of the neighbors of xby using the weighted average.
            phi_xyf = phi_interpolation(phi_xyf, phi_xyf.shape[0], self.weights[i][2], self.distance[i][
                2])  # We interpolate the phi values of the neighbors of xyf using the weighted average.
            phi_xyff = phi_interpolation(phi_xyff, phi_xyff.shape[0], self.weights[i][3], self.distance[i][
                3])  # We interpolate the phi values of the neighbors of xyff using the weighted average.

            # Second-order central diff in x-direction, and second-order forwards/ backward diff in y-direction.
            # Additionally, along y-direction, 2-order forward diff above flat plate and 2-order backward diff below
            # flat plate. This step is processed while creating the points. So, no need to distinguish
            # between the forward and backward diff.
            del_2_phi_x = (phi_xfy - 2 * phi_xy + phi_xby) / (self.cfg.custom.dx ** 2)
            del_2_phi_y = (phi_xyff - 2 * phi_xyf + phi_xy) / (self.cfg.custom.dy ** 2)

            # residual calculation within the belt
            grad_sq_phi_within_belt = del_2_phi_x + del_2_phi_y
            # Calculating u and v
            res_u_belt = u_pred - ((phi_xfy - phi_xby) / (2 * self.cfg.custom.dx))
            # We calculate the u value of the original point using the central difference formula.
            res_v_belt = v_pred - ((phi_xyf - phi_xy) / self.cfg.custom.dy)
            # We calculate the v value of the original point using the forward/backward difference formula.

            return grad_sq_phi_within_belt, res_u_belt, res_v_belt

        tpool = TPool()
        tpool.nthreads = int(os.popen('nproc').read())
        residuals_sort = tpool.map(residuals_calc, range(len(self.neighbors)))

        return torch.tensor(residuals_sort)

    @staticmethod
    def interior_total_losses(lambda_weigh_total, pred_values, true_outvar, invar, step):
        losses = {}

        for key, value in pred_values.items():
            _l = lambda_weigh_total * (torch.abs(pred_values[key] - true_outvar[key]).pow(2)) * invar["area"]
            losses[key] = _l.sum()

        # print(rf"At step = {step}, internal_loss = {losses}")
        return losses

    @torch.no_grad()
    def forward(self, invar, pre_outvar, true_outvar, lambda_weigh, step: int):

        if self.init_flag == 0:
            self.weights, self.neighbors, self.distance, self.belt_total = init_domain(invar, self.domain, self.cfg)
            self.init_flag = 1

        # Residual calc for all points within belt
        residuals = self.in_belt_residuals()

        # Assigning to corr variables
        grad_sq_phi_within_belt = residuals[:, [0]]
        res_u_belt = residuals[:, [1]]
        res_v_belt = residuals[:, [2]]

        # u, v and del_sq_phi values for each point within the belt are obtained. find the interior points which are
        # outside the belt and pull the residual_u, residual_v, and Poisson_2D values of them. Pull all the interior
        # points (excluding the wake and flat plate points) using invar variable.

        # interior points excluding wakes and flat plate points.
        total_interior = torch.hstack([invar['x'], invar['y']])
        self.belt_total = self.belt_total.to(self.device)

        # index of belt points present in interior
        index_belt_interior = get_index(self.belt_total, total_interior)

        # Segregating the residual values of interior points (excluding wakes and flat plate)

        residual_poisson_2d_belt = grad_sq_phi_within_belt[index_belt_interior]
        residual_u_belt = res_u_belt[index_belt_interior]
        residual_v_belt = res_v_belt[index_belt_interior]

        # belt total contains all wakes and interior points. First the interior points are seperated using
        # the index method.
        index_sort = pull_sort_wrt_x(self.belt_total, index_belt_interior)

        res_u_belt_sort = residual_u_belt[index_sort].to(self.device)
        res_v_belt_sort = residual_v_belt[index_sort].to(self.device)
        res_poiss_2d_belt_sort = residual_poisson_2d_belt[index_sort].to(self.device)

        # Similar step is carried out for total_interior points within belt and corr lambda values are sorted.
        # index of interior points present inside belt
        index_int_in_belt = get_index(total_interior, self.belt_total)

        # lambda weight in belt
        lambda_within_belt = lambda_weigh['Poisson_2D'][index_int_in_belt]
        index = pull_sort_wrt_x(total_interior, index_int_in_belt)

        # lambda values sorted wrt x-coordinates of total interior/invar.
        lambda_within_belt_sort = lambda_within_belt[index]

        # Predicted values outside belt.
        ######################
        # Since lambdas are function of co-ordinates points, but not on the 'Poisson_2D', 'residual_u' and 'residual_v'
        # any lambdas of all pred_outvar('Poisson_2D', 'residual_u' and 'residual_v') remains the same. Taking any one
        # for further calculation is fine.

        # index of interior points present outside belt
        index_out_belt = ~index_int_in_belt
        # lambda weigh out belt
        lambda_outside_belt = lambda_weigh['Poisson_2D'][index_out_belt]

        # residual values outside belt
        res_u_out_belt = pre_outvar['residual_u'][index_out_belt]
        res_v_out_belt = pre_outvar['residual_v'][index_out_belt]
        grad_sq_phi_outside = pre_outvar['Poisson_2D'][index_out_belt]
        ######################

        # total lambda
        lambda_weigh_total = torch.concat([lambda_outside_belt, lambda_within_belt_sort])

        # total residual
        pred_values = {'Poisson_2D': torch.concat([grad_sq_phi_outside, res_poiss_2d_belt_sort]),
                       'residual_u': torch.concat([res_u_out_belt, res_u_belt_sort]),
                       'residual_v': torch.concat([res_v_out_belt, res_v_belt_sort])
                       }
        # print(rf"alpha_interior: {invar['alpha'].unique()}")
        # print(rf"alpha_leftwall: {self.domain.constraints['LeftWall'].dataset.invar['alpha'][0]}")
        # print(rf"alpha_same: {self.alpha_same}")
        return self.interior_total_losses(lambda_weigh_total, pred_values, true_outvar, invar, step)
