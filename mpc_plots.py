import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import numpy as np

from config import T_STEP, A_MAX, A_MIN, V_MAX, W_BAR_SEARCH, A_W

T = T_STEP
W_BAR = W_BAR_SEARCH

def apply_plot_style(ax, title):
    ax.set_facecolor("#1a1a1a")
    ax.set_title(title, color="#ccc", fontsize=10)
    for sp in ax.spines.values():
        sp.set_color("#333")
    ax.tick_params(colors="#666", labelsize=8)
    ax.grid(color="#252525", linewidth=0.4)
    ax.xaxis.label.set_color("#888")
    ax.yaxis.label.set_color("#888")

def plot_path(sim, config, drone_path_out, drone_path_ret, truck_path, save="mpc_path.png", planner="A*"):
    cm = config["cell_metres"]
    gs = config["grid_size"]
    lx = sim["log_x"]
    phases = np.array(sim["log_phase"])

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#0f0f0f")
    apply_plot_style(ax, f"Drone round trip path — {planner}")

    for (x0, y0, w, h) in config["drone_restricted"]:
        ax.add_patch(patches.Rectangle((x0 * cm, y0 * cm), w * cm, h * cm, lw=0, fc="#8B0000", alpha=0.35))
    for (x0, y0, w, h) in config["truck_restricted"]:
        ax.add_patch(patches.Rectangle((x0 * cm, y0 * cm), w * cm, h * cm, lw=0, fc="#5C2E00", alpha=0.35))

    out_m = np.array(drone_path_out, dtype=float) * cm
    ret_m = np.array(drone_path_ret, dtype=float) * cm
    truck_m = np.array(truck_path, dtype=float) * cm
    ax.plot(out_m[:, 0], out_m[:, 1], "--", color="#4FC3F7", lw=1.0, alpha=0.5, label=f"{planner} outbound")
    ax.plot(ret_m[:, 0], ret_m[:, 1], "--", color="#81C784", lw=1.0, alpha=0.5, label=f"{planner} return")
    ax.plot(truck_m[:, 0], truck_m[:, 1], "--", color="#FFB74D", lw=0.7, alpha=0.4, label="Truck path")

    for ph, col in [("outbound", "#4FC3F7"), ("return", "#81C784")]:
        idx = np.where(phases == ph)[0]
        if len(idx):
            ax.plot(lx[idx, 0], lx[idx, 1], color=col, lw=2.2, label=f"MPC {ph}")

    launch_m = np.array(truck_path[sim["K_L_star"]], dtype=float) * cm
    delivery_m = np.array(config["drone_delivery"], dtype=float) * cm
    start_m = np.array(config["truck_start"], dtype=float) * cm

    ax.scatter(*start_m, s=80, c="#FFB74D", marker="s", zorder=6, label="Truck start")
    ax.scatter(*launch_m, s=150, c="#FFD700", marker="^", zorder=7, label=f"Launch (K*={sim['K_L_star']})")
    ax.scatter(*delivery_m, s=160, c="#E040FB", marker="*", zorder=7, label="Delivery point")
    if sim["k_landed"] is not None and sim["k_landed"] < len(lx):
        ax.scatter(*lx[sim["k_landed"], 0:2], s=150, c="#00E676", marker="v", zorder=8, label=f"Landed (step {sim['k_landed']})")
    ax.set_xlim(0, gs * cm)
    ax.set_ylim(0, gs * cm)
    ax.set_aspect("equal")
    ax.set_xlabel("x(m)")
    ax.set_ylabel("y(m)")
    ax.legend(loc="upper right", fontsize=8, facecolor="#222", labelcolor="#aaa", framealpha=0.8)
    plt.tight_layout()
    plt.savefig(save, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {save}")
    plt.close()


def plot_controls(sim, save="mpc_controls.png"):
    log_u = sim["log_u"]
    phases = np.array(sim["log_phase"])
    T_arr = np.arange(len(log_u)) * T

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor("#0f0f0f")
    fig.suptitle("MPC Control Inputs", color="#ccc", fontsize=12)

    labels = ["ax: drone acceleration(x) (m/s^2)", "ay: drone acceleration(y) (m/s^2)"]
    colors = ["#4FC3F7", "#81C784"]

    out_idx = np.where(phases == "outbound")[0]
    ret_idx = np.where(phases == "return")[0]

    for i, (ax, label, col) in enumerate(zip(axes, labels, colors)):
        apply_plot_style(ax, label)
        ax.plot(T_arr, log_u[:, i], color=col, lw=1.3)
        ax.axhline(A_MAX, color="#FF7043", lw=0.8, ls="--", alpha=0.5, label=f"±{A_MAX} m/s^2 limit")
        ax.axhline(A_MIN, color="#FF7043", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(0, color="#444", lw=0.5)
        if len(out_idx):
            ax.axvspan(T_arr[out_idx[0]], T_arr[out_idx[-1]], alpha=0.07, color="#4FC3F7", label="outbound")
        if len(ret_idx):
            ax.axvspan(T_arr[ret_idx[0]], T_arr[ret_idx[-1]], alpha=0.07, color="#81C784", label="return")
        ax.set_ylabel("m/s^2")
        ax.legend(fontsize=7, facecolor="#222", labelcolor="#aaa", loc="upper right")
    axes[-1].set_xlabel("Time(s)")
    plt.tight_layout()
    plt.savefig(save, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {save}")
    plt.close()


def plot_energy(sim, save="mpc_energy.png"):
    en = sim["energy"]
    T_arr = np.arange(len(en["power"])) * T
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor("#0f0f0f")
    fig.suptitle("Drone Energy Profile", color="#ccc", fontsize=12)

    # mark delivery / mass drop on both subplots
    k_d = sim.get("k_delivered")
    for ax in axes:
        if k_d is not None:
            ax.axvline(k_d * T, color="#E040FB", lw=1.0, ls="--", alpha=0.7, label=f"Delivery / mass drop (t={k_d * T:.1f} s)")

    apply_plot_style(axes[0], "Mechanical power")
    axes[0].plot(T_arr, en["power"], color="#CE93D8", lw=1.3)
    axes[0].fill_between(T_arr, en["power"], alpha=0.2, color="#CE93D8")
    axes[0].set_ylabel("Power (W)")
    axes[0].legend(fontsize=7, facecolor="#222", labelcolor="#aaa", loc="upper right")

    apply_plot_style(axes[1], f"Cumulative energy (total = {en['total_energy_J']} J)")
    axes[1].plot(T_arr, en["energy_cumulative"], color="#F48FB1", lw=1.5)
    axes[1].fill_between(T_arr, en["energy_cumulative"], alpha=0.15, color="#F48FB1")
    axes[1].set_ylabel("Energy (J)")
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(save, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {save}")
    plt.close()


def plot_velocity(sim, save="mpc_velocity.png"):
    lx = sim["log_x"]
    T_arr = np.arange(len(lx)) * T
    speed = np.linalg.norm(lx[:, 2:4], axis=1)

    phases = np.array(sim["log_phase"] + [""] * (len(lx) - len(sim["log_phase"])))
    out_idx = np.where(phases == "outbound")[0]
    ret_idx = np.where(phases == "return")[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.patch.set_facecolor("#0f0f0f")
    fig.suptitle("Drone State over Time", color="#ccc", fontsize=12)

    specs = [
        ("Position x", lx[:, 0], "#4FC3F7", "x (m)"),
        ("Position y", lx[:, 1], "#81C784", "y (m)"),
        ("Speed", speed, "#FFB74D", "m/s"),
    ]

    for ax, (title, data, col, ylabel) in zip(axes, specs):
        apply_plot_style(ax, title)
        ax.plot(T_arr, data, color=col, lw=1.3)
        if len(out_idx):
            ax.axvspan(T_arr[out_idx[0]], T_arr[out_idx[-1]], alpha=0.06, color="#4FC3F7")
        if len(ret_idx):
            ax.axvspan(T_arr[ret_idx[0]], T_arr[ret_idx[-1]], alpha=0.06, color="#81C784")
        ax.set_ylabel(ylabel)

    axes[2].axhline(V_MAX, color="#FF7043", lw=0.8, ls="--", alpha=0.5, label=f"Speed limit {V_MAX} m/s")
    axes[2].legend(fontsize=7, facecolor="#222", labelcolor="#aaa")
    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(save, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {save}")
    plt.close()


def plot_timing(sim, t_truck, t_launch, t_out, t_ret, save="mpc_timing.png", planner="A*"):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f0f0f")
    apply_plot_style(ax, f"Execution Time per Pipeline Stage — {planner}")

    stages = ["Truck A*", "Launch\nsearch", f"Outbound\n{planner}", f"Return\n{planner}", "MPC matrix\nbuild", "MPC sim\ntotal"]
    times  = [t_truck, t_launch, t_out, t_ret, sim["t_build_ms"], sim["t_sim_ms"]]
    colors = ["#FFB74D", "#FFD700", "#4FC3F7", "#81C784", "#CE93D8", "#F48FB1"]

    bars = ax.bar(stages, times, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{val} ms", ha="center", va="bottom", color="#ccc", fontsize=9)

    ax.set_ylabel("Time (ms)")
    ax.set_ylim(0, max(times) * 1.18)
    plt.tight_layout()
    plt.savefig(save, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {save}")
    plt.close()


def plot_wind(sim, save="mpc_wind.png"):
    lx = np.array(sim["log_x"])
    phases = np.array(sim["log_phase"])
    T_arr = np.arange(len(lx)) * T

    wx = lx[:, 4]
    wy = lx[:, 5]
    mag = np.sqrt(wx**2 + wy**2)
    ang = np.degrees(np.arctan2(wy, wx))

    out_idx = np.where(phases == "outbound")[0]
    ret_idx = np.where(phases == "return")[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.patch.set_facecolor("#0f0f0f")
    fig.suptitle("Wind State (OU Process)", color="#ccc", fontsize=12)

    def shade(ax):
        if len(out_idx):
            ax.axvspan(T_arr[out_idx[0]], T_arr[out_idx[-1]], alpha=0.06, color="#4FC3F7")
        if len(ret_idx):
            ax.axvspan(T_arr[ret_idx[0]], T_arr[ret_idx[-1]], alpha=0.06, color="#81C784")

    apply_plot_style(axes[0], "Wind velocity components")
    axes[0].plot(T_arr, wx, color="#4FC3F7", lw=1.3, label="w_x")
    axes[0].plot(T_arr, wy, color="#81C784", lw=1.3, label="w_y")
    axes[0].axhline(W_BAR[0], color="#4FC3F7", lw=0.8, ls="--", alpha=0.5, label=f"w̄_x={W_BAR[0]}")
    axes[0].axhline(W_BAR[1], color="#81C784", lw=0.8, ls="--", alpha=0.5, label=f"w̄_y={W_BAR[1]}")
    axes[0].set_ylabel("m/s")
    axes[0].legend(fontsize=7, facecolor="#222", labelcolor="#aaa", loc="upper right")
    shade(axes[0])

    apply_plot_style(axes[1], "Wind magnitude")
    axes[1].plot(T_arr, mag, color="#FFB74D", lw=1.3)
    axes[1].axhline(np.linalg.norm(W_BAR), color="#FFB74D", lw=0.8, ls="--", alpha=0.5, label=f"|w̄|={np.linalg.norm(W_BAR):.2f} m/s")
    axes[1].set_ylabel("|w| (m/s)")
    axes[1].legend(fontsize=7, facecolor="#222", labelcolor="#aaa", loc="upper right")
    shade(axes[1])

    apply_plot_style(axes[2], "Wind direction (deg from +x)")
    axes[2].plot(T_arr, ang, color="#CE93D8", lw=1.3)
    axes[2].axhline(np.degrees(np.arctan2(W_BAR[1], W_BAR[0])), color="#CE93D8",
                    lw=0.8, ls="--", alpha=0.5,
                    label=f"mean dir={np.degrees(np.arctan2(W_BAR[1], W_BAR[0])):.1f}°")
    axes[2].set_ylabel("degrees")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=7, facecolor="#222", labelcolor="#aaa", loc="upper right")
    shade(axes[2])

    plt.tight_layout()
    plt.savefig(save, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {save}")
    plt.close()


def plot_gif(sim, config, drone_path_out, drone_path_ret, truck_path,
             save="mpc_animation.gif", planner="A*", stride=2, fps=15):
    cm = config["cell_metres"]
    gs = config["grid_size"]
    lx = np.array(sim["log_x"])
    phases = sim["log_phase"]
    K_L = sim["K_L_star"]
    k_del = sim["k_delivered"]
    k_land = sim["k_landed"]
    N = len(lx)

    truck_pos = lx[:, 6:8]
    drone_pos = lx[:, 0:2]
    wind_vec = lx[:, 4:6]

    out_m   = np.array(drone_path_out, dtype=float)*cm
    ret_m   = np.array(drone_path_ret, dtype=float)*cm
    truck_m = np.array(truck_path, dtype=float)*cm

    delivery_m = np.array(config["drone_delivery"], dtype=float) * cm

    frame_indices = list(range(0, N, stride))
    if (N - 1) not in frame_indices:
        frame_indices.append(N - 1)

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#0f0f0f")
    apply_plot_style(ax, f"Truck-Drone Delivery Animation — {planner}")

    for (x0, y0, w, h) in config["drone_restricted"]:
        ax.add_patch(patches.Rectangle((x0*cm, y0*cm), w*cm, h*cm,
                                       lw=0, fc="#8B0000", alpha=0.35))
    for (x0, y0, w, h) in config["truck_restricted"]:
        ax.add_patch(patches.Rectangle((x0*cm, y0*cm), w*cm, h*cm,
                                       lw=0, fc="#5C2E00", alpha=0.35))
    ax.plot(out_m[:,0],   out_m[:,1],   "--", color="#4FC3F7", lw=0.8, alpha=0.35, label="Planned outbound")
    ax.plot(ret_m[:,0],   ret_m[:,1],   "--", color="#81C784", lw=0.8, alpha=0.35, label="Planned return")
    ax.plot(truck_m[:,0], truck_m[:,1], "--", color="#FFB74D", lw=0.6, alpha=0.30, label="Truck road")
    ax.scatter(*delivery_m, s=160, c="#E040FB", marker="*", zorder=7, label="Delivery point")

    drone_trail, = ax.plot([], [], color="#4FC3F7", lw=1.6, alpha=0.7)

    drone_dot, = ax.plot([], [], "o", color="#00E5FF", ms=9, zorder=9, label="Drone")

    truck_dot, = ax.plot([], [], "s", color="#FFB74D", ms=10, zorder=9, label="Truck")

    wind_scale = 8.0
    wind_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                             arrowprops=dict(arrowstyle="->", color="#FF6F00", lw=1.5))

    time_text  = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                         color="#ccc", fontsize=9, va="top",
                         bbox=dict(boxstyle="round,pad=0.3", fc="#111", ec="#333", alpha=0.8))

    wind_label = ax.text(0.02, 0.89, "", transform=ax.transAxes,
                         color="#FF6F00", fontsize=8, va="top")

    ax.set_xlim(0, gs * cm)
    ax.set_ylim(0, gs * cm)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=7, facecolor="#222",
              labelcolor="#aaa", framealpha=0.8)

    def init():
        drone_trail.set_data([], [])
        drone_dot.set_data([], [])
        truck_dot.set_data([], [])
        wind_arrow.set_visible(False)
        time_text.set_text("")
        wind_label.set_text("")
        return drone_trail, drone_dot, truck_dot, wind_arrow, time_text, wind_label

    def update(frame_num):
        k = frame_indices[frame_num]
        t_now = k * T

        ph = phases[k] if k < len(phases) else phases[-1]

        if ph in ("prelaunch", "postlanding"):
            drone_trail.set_color("#AAAAAA")
            drone_trail.set_data([], [])
            dp = drone_pos[k]
            tp = truck_pos[k]
            drone_dot.set_data([dp[0]], [dp[1]])
            truck_dot.set_data([tp[0]], [tp[1]])
            wx, wy = wind_vec[k]
            arrow_end = dp + wind_scale * np.array([wx, wy])
            wind_arrow.xy = arrow_end
            wind_arrow.xytext = dp
            wind_arrow.set_visible(True)
            label = "pre-launch (on truck)" if ph == "prelaunch" else "post-landing (on truck)"
            time_text.set_text(f"t = {t_now:.1f} s  |  phase: {label}")
            wind_label.set_text(f"wind: ({wx:.2f}, {wy:.2f}) m/s  "
                                f"|w|={np.hypot(wx,wy):.2f}")
        else:
            trail_col = "#4FC3F7" if ph == "outbound" else "#81C784"
            drone_trail.set_color(trail_col)

            trail_start = max(K_L, k - 80)
            drone_trail.set_data(drone_pos[trail_start:k+1, 0],
                                  drone_pos[trail_start:k+1, 1])

            dp = drone_pos[k]
            tp = truck_pos[k]
            wx, wy = wind_vec[k]

            drone_dot.set_data([dp[0]], [dp[1]])
            truck_dot.set_data([tp[0]], [tp[1]])

            arrow_end = dp + wind_scale * np.array([wx, wy])
            wind_arrow.xy = arrow_end
            wind_arrow.xytext = dp
            wind_arrow.set_visible(True)

            landed_marker    = "  LANDED"        if (k_land is not None and k >= k_land) else ""
            delivered_marker = " [pkg delivered]" if (k_del  is not None and k >= k_del)  else ""
            time_text.set_text(
                f"t = {t_now:.1f} s | phase: {ph}{delivered_marker}{landed_marker}"
            )
            wind_label.set_text(f"wind: ({wx:.2f}, {wy:.2f}) m/s  "
                                f"|w|={np.hypot(wx,wy):.2f}")

        return drone_trail, drone_dot, truck_dot, wind_arrow, time_text, wind_label

    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_indices),
        init_func=init, blit=True, interval=1000 // fps
    )

    writer = animation.PillowWriter(fps=fps)
    ani.save(save, writer=writer, dpi=100)
    print(f"Saved: {save}")
    plt.close()
