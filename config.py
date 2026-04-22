import numpy as np
TRUCK_MARGIN = 2
DRONE_MARGIN = 14
TRUCK_SPEED = 2
SAMPLE_S = 10 # search interval for K_L_star
T_STEP = 0.5 # time step duration in seconds
V_CRUISE = 10.0 # drone speed for search (must be <= V_MAX)
DRONE_WEIGHT_TO_TOTAL_WEIGHT_RATIO = 0.75 # package drop parameter
W_BAR_SEARCH = np.array([1.6, 0.9])
A_W = 0.4  # OU wind mean reversion rate
SIGMA_OU = 0.3  # stochastic noise std

F = 15
MASS = 1.0
A_MAX = 5.0
A_MIN = -5.0
V_MAX = 12.0
V_MIN = -12.0

TOL_DELIVERY = 10.0
TOL_LANDING = 15.0

Q = np.diag([300.0, 300.0, 3.0, 3.0])
R = np.diag([0.5, 0.5])
Qf = 5.0 * Q