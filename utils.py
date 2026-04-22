import numpy as np
from config import V_CRUISE, W_BAR_SEARCH
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def format_seconds(step_count, T):
        if step_count is None:
            return "not reached"
        else:
            return f"{step_count*T}s"

def format_airtime(sim, T):
    if sim["k_landed"] is None or sim["K_launch_step"] is None:
        return "not available"
    else:
        return f"{(sim['k_landed']-sim['K_launch_step'])*T}s"

def print_planner_stats(name, sim, path_out, path_ret, T):
    print()
    print(f"{name} summary")
    print(f"- Total path length: {len(path_out)+len(path_ret)} cells")
    print(f"- MPC simulation time: {sim['t_sim_ms']} ms")
    print(f"- Delivery time: {format_seconds(sim['k_delivered'], T)}")
    print(f"- Landing time: {format_seconds(sim['k_landed'], T)}")
    print(f"- Airtime: {format_airtime(sim, T)}")
    print(f"- Total energy: {sim['energy']['total_energy_J']} J")
    print()


def wind_adjusted_speed(start_m, end_m):
    vec  = np.asarray(end_m) - np.asarray(start_m)
    dist = np.linalg.norm(vec)
    if dist < 1e-6:
        return V_CRUISE
    return max(V_CRUISE + float(np.dot(W_BAR_SEARCH, vec / dist)), 0)


def visualise_occupancy_grid(truck_grid, drone_grid, config, save_path="step1_map.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0f0f0f')
    titles = ["Truck occupancy grid", "Drone occupancy grid"]
    grids  = [truck_grid, drone_grid]
    zones  = [config["truck_restricted"], config["drone_restricted"]]
    colors = ['#8B4513', '#8B0000']

    for ax, grid, title, zone_list, zcolor in zip(axes, grids, titles, zones, colors):
        ax.set_facecolor('#1a1a1a')
        ax.set_title(title, color='#ccc', fontsize=11)
        ax.tick_params(colors='#666', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#333')

        display = np.where(grid, 0.9, 0.2)
        ax.imshow(display, origin='lower', cmap='gray', vmin=0, vmax=1, aspect='equal')

        for (xmin, ymin, w, h) in zone_list:
            rect = patches.Rectangle((xmin, ymin), w, h, linewidth=1, edgecolor=zcolor, facecolor=zcolor, alpha=0.4)
            ax.add_patch(rect)
        gs = config["grid_size"]
        ts = config["truck_start"]
        ax.scatter(*ts, s=80, c='#FFB74D', zorder=5, marker='s', label='Truck start')
        te = config["truck_end"]
        ax.scatter(*te, s=80, c='#FF7043', zorder=5, marker='s', label='Truck end')
        dd = config["drone_delivery"] 
        ax.scatter(*dd, s=120,c='#E040FB', zorder=5, marker='*', label='Delivery')

        ax.set_xlim(0, gs); ax.set_ylim(0, gs)
        ax.set_xlabel("x (cells)", color='#888')
        ax.set_ylabel("y (cells)", color='#888')
        ax.legend(loc='upper right', fontsize=7, facecolor='#222', labelcolor='#aaa', framealpha=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Visualisation saved to {save_path}")
    plt.close()