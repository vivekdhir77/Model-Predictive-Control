import time
import numpy as np
from scipy.linalg import solve_discrete_are
from config import (T_STEP, TRUCK_SPEED, V_CRUISE, F, MASS, A_MAX, A_MIN, V_MAX, V_MIN, TOL_DELIVERY, TOL_LANDING, W_BAR_SEARCH, SIGMA_OU)
from init_path import A_star_path, wind_biased_rrt, estimate_truck_position, gamma, launch_search
from map import load_map
from mpc_plots import plot_path, plot_controls, plot_energy, plot_velocity, plot_timing, plot_wind, plot_gif
from config import DRONE_WEIGHT_TO_TOTAL_WEIGHT_RATIO
from utils import print_planner_stats
T = T_STEP


A_W = 0.4
W_BAR = W_BAR_SEARCH
NX = 10
NU = 4
AD = np.zeros((6, 6))
AD[0, 0]= 1
AD[0, 2]= T
AD[1, 1]= 1
AD[1, 3]= T
AD[2, 2]= 1
AD[2, 4]= T
AD[3, 3]= 1
AD[3, 5]= T
AD[4, 4]= 1-A_W*T
AD[5, 5]= 1-A_W*T

BD = np.zeros((6, 2))
BD[0, 0] = 0.5*T**2
BD[1, 1] = 0.5*T**2
BD[2, 0] = T
BD[3, 1] = T

dD = np.zeros(6)
dD[4] = A_W*T*W_BAR[0]
dD[5] = A_W*T*W_BAR[1]

AT = np.eye(4)
AT[0, 2] = T
AT[1, 3] = T

BT = np.zeros((4, 2))
BT[0, 0] = 0.5*T**2;
BT[1, 1] = 0.5*T**2
BT[2, 0] = T
BT[3, 1] = T

A_SYS = np.zeros((NX, NX))
A_SYS[:6, :6] = AD
A_SYS[6:, 6:] = AT

B_SYS = np.zeros((NX, NU))
B_SYS[:6, :2] = BD
B_SYS[6:, 2:] = BT

D_SYS = np.zeros(NX)
D_SYS[:6] = dD

Q_STAGE = np.diag([500.0, 500.0, 5.0, 5.0, 0.0, 0.0, 30.0, 30.0, 1.0, 1.0])
R_CTRL = np.diag([0.1, 0.1, 0.3, 0.3])

try:
    Q_TERM = solve_discrete_are(A_SYS, B_SYS, Q_STAGE, R_CTRL)
except Exception:
    Q_TERM = 5.0 * Q_STAGE

def build_prediction_matrices():
    O = np.zeros((F * NX, NX))
    Mm = np.zeros((F * NX, F * NU))
    D_lift = np.zeros((F * NX, NX))
    Ak = np.eye(NX)
    D_cum = np.zeros(NX)
    for i in range(F):
        Ak = A_SYS @ Ak
        D_cum = A_SYS @ D_cum + D_SYS
        O[NX*i: NX*(i+1), :] = Ak
        D_lift[NX*i: NX*(i+1), :] = np.outer(D_cum, np.ones(NX)) / NX
        for j in range(i + 1):
            Mm[NX*i:NX*(i+1), NU*j:NU*(j+1)] = np.linalg.matrix_power(A_SYS, i - j) @ B_SYS
    D_vec = np.zeros(F * NX)
    D_acc = np.zeros(NX)
    for i in range(F):
        D_acc = A_SYS @ D_acc + D_SYS
        D_vec[NX*i:NX*(i+1)] = D_acc
    return O, Mm, D_vec


def build_cost_matrices(Mm):
    W4 = np.zeros((F * NX, F * NX))
    for i in range(F - 1):
        W4[NX*i:NX*(i+1), NX*i:NX*(i+1)] = Q_STAGE
    W4[NX*(F-1):NX*F, NX*(F-1):NX*F] = Q_TERM

    W3 = np.zeros((F * NU, F * NU))
    for i in range(F):
        W3[NU*i:NU*(i+1), NU*i:NU*(i+1)] = R_CTRL

    H = 2.0 * (Mm.T @ W4 @ Mm + W3)
    return H, W4, W3


def solve_mpc(H, W4, Mm, O, D_vec, x0, X_ref, u_min, u_max, v_min, v_max):
    Sk = X_ref - O @ x0 - D_vec
    fk = -2.0 * Mm.T @ W4 @ Sk
    try:
        G = np.linalg.solve(H, -fk)
    except np.linalg.LinAlgError:
        G = np.linalg.lstsq(H, -fk, rcond=None)[0]
    U_opt = np.clip(G, u_min, u_max)
    return U_opt


def build_horizon_ref(pos_ref_drone, vel_ref_drone, pos_ref_truck, vel_ref_truck, drone_step, global_step=0, blend_truck=False):
    X_ref = np.zeros(F * NX)
    n_ret = len(pos_ref_drone)
    for i in range(F):
        idx_d = min(drone_step + i + 1, n_ret - 1)
        idx_t = min(global_step + i + 1, len(pos_ref_truck) - 1)
        base = NX * i

        truck_pos = pos_ref_truck[idx_t]
        truck_vel = vel_ref_truck[idx_t]

        if blend_truck:
            progress = drone_step / max(n_ret - 1, 1)
            alpha = min((progress - 0.9) / 0.1, 1.0)
            drone_pos = (1-alpha)*pos_ref_drone[idx_d]+alpha * truck_pos
            drone_vel = (1-alpha)*vel_ref_drone[idx_d]+alpha * truck_vel
        else:
            drone_pos = pos_ref_drone[idx_d]
            drone_vel = vel_ref_drone[idx_d]
        X_ref[base:base+2] = drone_pos
        X_ref[base+2:base+4] = drone_vel
        X_ref[base+4:base+6] = W_BAR
        X_ref[base+6:base+8] = truck_pos
        X_ref[base+8:base+10] = truck_vel
    return X_ref

def time_parameterise_truck(truck_path_cells, speed, cell_metres, n_steps):
    pts = np.array(truck_path_cells, dtype=float) * cell_metres
    diffs = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = arc[-1]
    pos_ref = np.zeros((n_steps, 2))
    vel_ref = np.zeros((n_steps, 2))
    for k in range(n_steps):
        s = min(k * speed * T, total)
        idx = int(np.searchsorted(arc, s, side="right")) - 1
        idx = np.clip(idx, 0, len(diffs) - 1)
        t_frac = (s - arc[idx]) / max(seg_len[idx], 1e-9)
        pos_ref[k] = pts[idx] + t_frac * diffs[idx]
        vel_ref[k] = diffs[idx] / max(seg_len[idx], 1e-9) * speed
    return pos_ref, vel_ref


def time_parameterise(path_cells, speed, cell_metres):
    pts = np.array(path_cells, dtype=float) * cell_metres
    diffs = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = arc[-1]
    n_steps = max(int(total/(speed*T))+2, 2)
    pos_ref = np.zeros((n_steps, 2))
    vel_ref = np.zeros((n_steps, 2))
    for k in range(n_steps):
        s = min(k * speed * T, total)
        idx = int(np.searchsorted(arc, s, side="right")) - 1
        idx = np.clip(idx, 0, len(diffs) - 1)
        t = (s - arc[idx]) / max(seg_len[idx], 1e-9)
        pos_ref[k] = pts[idx] + t*diffs[idx]
        vel_ref[k] = diffs[idx]/max(seg_len[idx], 1e-9) * speed
    return pos_ref, vel_ref

def compute_energy(log_x, log_u, log_mass):
    u_arr = np.array(log_u)
    x_arr = np.array(log_x)[:len(u_arr)]
    m_arr = np.array(log_mass)[:len(u_arr)]
    force = m_arr[:, None] * u_arr
    velocity = x_arr[:, 2:4]
    power = np.sum(force * velocity, axis=1)
    power = np.maximum(power, 0.0)
    energy_cumulative = np.cumsum(power) * T
    total_energy = float(energy_cumulative[-1]) if len(energy_cumulative) else 0.0
    return {"power": power, "energy_cumulative": energy_cumulative, "total_energy_J": total_energy}

def run_mpc(config, truck_path, K_L_star, drone_path_out, drone_path_ret):
    cm = config["cell_metres"]
    t0 = time.time()
    O, Mm, D_vec = build_prediction_matrices()
    H, W4, W3 = build_cost_matrices(Mm)
    t_build_ms = (time.time() - t0) * 1000
    print(f"MPC setup finished in {t_build_ms} ms.")

    pos_out, vel_out = time_parameterise(drone_path_out, V_CRUISE, cm)
    pos_ret, vel_ret = time_parameterise(drone_path_ret, V_CRUISE, cm)

    truck_arc = gamma(truck_path)
    dist_to_launch = truck_arc[K_L_star] * cm
    K_launch_step = int(round(dist_to_launch / (TRUCK_SPEED * T)))
    print(f"Launch point (K_L_star): {K_L_star}, which lines up with step {K_launch_step} ({K_launch_step * T} s)")

    dist_truck_total = truck_arc[len(truck_path) - 1] * cm
    K_truck_end_step = int(round(dist_truck_total / (TRUCK_SPEED * T)))
    print(f"The truck reaches its destination at step {K_truck_end_step} ({K_truck_end_step * T} s)")

    max_drone_steps = len(pos_out) + len(pos_ret) + 20
    total_steps = max(K_truck_end_step, K_launch_step + max_drone_steps) + 50
    pos_truck, vel_truck = time_parameterise_truck(
        truck_path, TRUCK_SPEED, cm, total_steps + F + 1)

    log_x = []
    log_u = []
    log_phase = []
    log_mass = [MASS]

    print(f"Simulating the pre-launch segment for {K_launch_step} steps while the truck drives to the handoff point")

    for k in range(K_launch_step):
        elapsed_m = k * T * TRUCK_SPEED
        truck_cell = estimate_truck_position(truck_arc, truck_path, elapsed_m / cm)
        truck_m = np.array(truck_cell, dtype=float) * cm

        x_pre = np.zeros(NX)
        x_pre[0:2] = truck_m
        x_pre[2:4] = np.array([TRUCK_SPEED, 0.0])
        if k == 0:
            x_pre[4:6] = W_BAR
        else:
            prev_w = log_x[-1][4:6]
            noise = SIGMA_OU * np.sqrt(T) * np.random.randn(2)
            x_pre[4:6] = (1 - A_W*T)*prev_w + A_W*T*W_BAR + noise
        x_pre[6:8] = truck_m
        x_pre[8:10] = np.array([TRUCK_SPEED, 0.0])

        log_x.append(x_pre.copy())
        log_u.append(np.zeros(2))
        log_phase.append("prelaunch")
        log_mass.append(MASS)
    launch_m = np.array(truck_path[K_L_star], dtype=float) * cm
    delivery_m = np.array(config["drone_delivery"], dtype=float) * cm

    x = np.zeros(NX)
    x[0:2] = launch_m
    x[2:4] = np.array([TRUCK_SPEED, 0.0])
    x[4:6] = log_x[-1][4:6] if log_x else W_BAR
    x[6:8] = launch_m
    x[8:10] = np.array([TRUCK_SPEED, 0.0])
    log_x.append(x.copy())

    u_min = np.full(F * NU, A_MIN)
    u_max = np.full(F * NU, A_MAX)
    v_min = V_MIN
    v_max = V_MAX

    mass = MASS
    phase = "outbound"
    step_out = 0
    step_ret = 0
    k_delivered = None
    k_landed = None
    k_truck_end = None
    t_sim_start = time.time()
    truck_end_m = np.array(truck_path[-1], dtype=float) * cm

    for step in range(total_steps):
        abs_step = K_launch_step + step
        drone_pos = x[0:2]
        truck_pos = x[6:8]
        if phase == "outbound":
            if np.linalg.norm(drone_pos - delivery_m) < TOL_DELIVERY:
                phase = "return"
                k_delivered = abs_step
                mass = MASS * DRONE_WEIGHT_TO_TOTAL_WEIGHT_RATIO
                print(f"Delivery completed at step {k_delivered} ({k_delivered * T} s). Drone mass drops from {MASS} kg to {mass} kg.")

        elif phase == "return":
            if np.linalg.norm(drone_pos - truck_pos) < TOL_LANDING:
                k_landed = abs_step
                phase = "postlanding"
                print(f"The drone lands back on the truck at step {k_landed} ({k_landed * T} s).")

        elif phase == "postlanding":
            if np.linalg.norm(truck_pos - truck_end_m) < TOL_LANDING:
                k_truck_end = abs_step
                print(f"The truck finishes the route at step {k_truck_end} ({k_truck_end * T} s).")
                x[0:2] = truck_end_m
                x[6:8] = truck_end_m
                log_x.append(x.copy())
                log_u.append(np.zeros(2))
                log_phase.append("postlanding")
                log_mass.append(mass)
                break

        if phase in ("outbound", "return"):
            global_step = abs_step
            if phase == "outbound":
                X_ref = build_horizon_ref(pos_out, vel_out, pos_truck, vel_truck, step_out, global_step)
                step_out = min(step_out + 1, len(pos_out) - 1)
            else:
                blend = (step_ret / max(len(pos_ret) - 1, 1)) >= 0.9
                X_ref = build_horizon_ref(pos_ret, vel_ret, pos_truck, vel_truck, step_ret, global_step, blend)
                step_ret = min(step_ret + 1, len(pos_ret) - 1)

            t_qp = time.time()
            U_opt = solve_mpc(H, W4, Mm, O, D_vec, x, X_ref, u_min, u_max, v_min, v_max)
            u_drone = np.clip(U_opt[0:2], A_MIN, A_MAX)
            u_truck = np.clip(U_opt[2:4], A_MIN, A_MAX)
            dt_ms = (time.time() - t_qp) * 1000
        else:
            u_drone = np.zeros(2)
            u_truck = np.zeros(2)
            dt_ms = 0.0

        u_full = np.concatenate([u_drone, u_truck])
        x = A_SYS @ x + B_SYS @ u_full + D_SYS
        x[4:6] += SIGMA_OU * np.sqrt(T) * np.random.randn(2)
        x[2] = np.clip(x[2], V_MIN, V_MAX)
        x[3] = np.clip(x[3], V_MIN, V_MAX)

        elapsed_m = abs_step * T * TRUCK_SPEED
        truck_cell = estimate_truck_position(truck_arc, truck_path, elapsed_m / cm)
        truck_m = np.array(truck_cell, dtype=float) * cm
        x[6:8] = truck_m
        x[8:10] = np.array([TRUCK_SPEED, 0.0])

        if phase == "postlanding":
            x[0:2] = truck_m
            x[2:4] = np.array([TRUCK_SPEED, 0.0])

        log_x.append(x.copy())
        log_u.append(u_drone.copy())
        log_phase.append(phase)
        log_mass.append(mass)

        if step % 50 == 0:
            print(f"Step {abs_step} | {phase}")
            print(f"drone=({x[0]}, {x[1]})")
            print(f"truck=({x[6]}, {x[7]})")
            print(f"wind=({x[4]}, {x[5]})")
            print(f"solve={dt_ms} ms")

    t_sim_ms = (time.time() - t_sim_start) * 1000

    flight_log_u = [u for u, ph in zip(log_u, log_phase) if ph != "prelaunch"]
    flight_log_x = [x for x, ph in zip(log_x, log_phase) if ph != "prelaunch"]
    flight_log_m = [m for m, ph in zip(log_mass[1:], log_phase) if ph != "prelaunch"]

    log_x = np.array(log_x)
    log_u = np.array(log_u)
    log_mass = np.array(log_mass[1:])
    energy = compute_energy(np.array(flight_log_x), np.array(flight_log_u) if flight_log_u else np.zeros((1, 2)), np.array(flight_log_m) if flight_log_m else np.array([MASS]))

    log_dt_ms = np.zeros(max(1, len(flight_log_u)))

    print("\nSimulation summary")
    print(f"- MPC setup time: {t_build_ms}ms")
    print(f"- Total simulation time: {t_sim_ms}ms")

    if k_delivered is not None:
        print(f"- Delivery reached at step {k_delivered} ({k_delivered*T}s)")
    else:
        print("- Delivery was not reached.")

    if k_landed is not None:
        print(f"- Drone landed at step {k_landed} ({k_landed * T}s)")
    else:
        print("- The drone did not land on the truck.")

    if k_truck_end is not None:
        print(f"- Truck finished at step {k_truck_end} ({k_truck_end*T}s)")
    else:
        print("- The truck did not reach the end of the route.")

    if k_delivered is not None and k_landed is not None:
        airtime = (k_landed - K_launch_step)*T
        print(f"- Drone airtime after launch: {airtime}s")
    else:
        print("- The drone did not land on the truck.")

    print(f"- Total energy used: {energy['total_energy_J']}J")

    return {
        "log_x": log_x,
        "log_u": log_u,
        "log_phase": log_phase,
        "log_mass": log_mass,
        "log_dt_ms": log_dt_ms,
        "energy": energy,
        "k_delivered": k_delivered,
        "k_landed": k_landed,
        "k_truck_end": k_truck_end,
        "K_L_star": K_L_star,
        "K_launch_step": K_launch_step,
        "t_build_ms": t_build_ms,
        "t_sim_ms": t_sim_ms,
    }

def steps_to_s(k): 
    if k is None:
        return "None"
    else:
        return f"{k * T} s"

if __name__ == "__main__":
    print("MPC drone simulation")
    print("Comparing A*, wind-biased RRT, and standard RRT.\n")

    truck_grid, drone_grid, config = load_map("map.npz")
    cm = config["cell_metres"]
    print("Map loaded")

    NPZ = "init_path_results.npz"
    try:
        data = np.load(NPZ, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"{NPZ} not found. Run init_path.py first to generate it.")
    print(f"Loaded precomputed paths from {NPZ}:")

    K_L_star = int(data["K_L_star"])
    truck_path = [tuple(c) for c in data["truck_path"]]
    astar_out = [tuple(c) for c in data["drone_path_1_star"]]
    astar_ret = [tuple(c) for c in data["drone_path_2_star"]]
    rrt_out = [tuple(c) for c in data["drone_path_1_rrt"]]
    rrt_ret = [tuple(c) for c in data["drone_path_2_rrt"]]
    std_rrt_out = [tuple(c) for c in data["drone_path_1_std_rrt"]]
    std_rrt_ret = [tuple(c) for c in data["drone_path_2_std_rrt"]]

    print(f"- Launch index: {K_L_star}")
    print(f"- Truck path: {len(truck_path)} cells")
    print(f"- A* outbound: {len(astar_out)} cells")
    print(f"- A* return: {len(astar_ret)} cells")
    print(f"- Wind-biased RRT outbound: {len(rrt_out)} cells")
    print(f"- Wind-biased RRT return: {len(rrt_ret)} cells")
    print(f"- Standard RRT outbound: {len(std_rrt_out)} cells")
    print(f"- Standard RRT return: {len(std_rrt_ret)} cells")

    t_truck = t_launch = t_astar_out = t_astar_ret = t_rrt_out = t_rrt_ret = t_std_rrt_out = t_std_rrt_ret = 0.0

    print("Running the A* scenario:\n")
    sim_astar = run_mpc(config, truck_path, K_L_star, astar_out, astar_ret)

    plot_path(sim_astar, config, astar_out, astar_ret, truck_path, save="mpc_path_astar.png", planner="A*")
    plot_controls(sim_astar, save="mpc_controls_astar.png")
    plot_energy(sim_astar, save="mpc_energy_astar.png")
    plot_velocity(sim_astar, save="mpc_velocity_astar.png")
    plot_timing(sim_astar, t_truck, t_launch, t_astar_out, t_astar_ret, save="mpc_timing_astar.png", planner="A*")
    plot_wind(sim_astar, save="mpc_wind_astar.png")
    plot_gif(sim_astar, config, astar_out, astar_ret, truck_path, save="mpc_animation_astar.gif", planner="A*")
    print("Saved the A* plots.")

    print("Running the wind-biased RRT scenario:\n")
    sim_rrt = run_mpc(config, truck_path, K_L_star, rrt_out, rrt_ret)

    plot_path(sim_rrt, config, rrt_out, rrt_ret, truck_path, save="mpc_path_rrt.png", planner="Wind RRT")
    plot_controls(sim_rrt, save="mpc_controls_rrt.png")
    plot_energy(sim_rrt, save="mpc_energy_rrt.png")
    plot_velocity(sim_rrt, save="mpc_velocity_rrt.png")
    plot_timing(sim_rrt, t_truck, t_launch, t_rrt_out, t_rrt_ret, save="mpc_timing_rrt.png", planner="Wind RRT")
    plot_wind(sim_rrt, save="mpc_wind_rrt.png")
    plot_gif(sim_rrt, config, rrt_out, rrt_ret, truck_path, save="mpc_animation_rrt.gif", planner="Wind RRT")
    print("Saved the wind-biased RRT plots.")

    print("Running the standard RRT scenario:\n")
    sim_std_rrt = run_mpc(config, truck_path, K_L_star, std_rrt_out, std_rrt_ret)

    plot_path(sim_std_rrt, config, std_rrt_out, std_rrt_ret, truck_path, save="mpc_path_std_rrt.png", planner="Std RRT")
    plot_controls(sim_std_rrt, save="mpc_controls_std_rrt.png")
    plot_energy(sim_std_rrt, save="mpc_energy_std_rrt.png")
    plot_velocity(sim_std_rrt, save="mpc_velocity_std_rrt.png")
    plot_timing(sim_std_rrt, t_truck, t_launch, t_std_rrt_out, t_std_rrt_ret, save="mpc_timing_std_rrt.png", planner="Std RRT")
    plot_wind(sim_std_rrt, save="mpc_wind_std_rrt.png")
    plot_gif(sim_std_rrt, config, std_rrt_out, std_rrt_ret, truck_path, save="mpc_animation_std_rrt.gif", planner="Std RRT")
    print("Saved the standard RRT plots.")


    print("\nPlanner comparison")
    print_planner_stats("A*", sim_astar, astar_out, astar_ret, T)
    print_planner_stats("Wind-biased RRT", sim_rrt, rrt_out, rrt_ret, T)
    print_planner_stats("Standard RRT", sim_std_rrt, std_rrt_out, std_rrt_ret, T)
