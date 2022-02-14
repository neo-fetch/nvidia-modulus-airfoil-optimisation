import math
import numpy as np
# We create component velocities u and v such that sqrt(u^2 + v^2) = 1 and tan(theta) = u/v, where theta is in degrees.

def get_angle(theta):
    tan = math.tan(theta)
    u = math.sqrt(1/(1+tan**2))
    v = u*tan
    return u, v

if __name__ == '__main__':
# Getting u and v from theta ranging from -10 to 10 degrees with increments of 0.01 degrees.
    for theta in np.arange(-10, 10.01, 0.01):
        u, v = get_angle(theta*math.pi/180)
        # print(f"{theta:.2f} degrees: u = {u:.6f}, v = {v:.6f}")
        print(f"{u:.6f} {v:.6f}")
