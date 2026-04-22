# Map Creation
import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from map_config import map_config
from config import TRUCK_MARGIN, DRONE_MARGIN
from utils import visualise_occupancy_grid

def create_occupancy_grid(restricted_list, grid_size, margin):
    grid = np.ones((grid_size, grid_size), dtype=bool)
    for (xmin, ymin, w, h) in restricted_list:
        r0 = max(0, ymin - margin)
        r1 = min(grid_size, ymin+h+margin)
        c0 = max(0, xmin - margin)
        c1 = min(grid_size, xmin+w+margin)
        grid[r0:r1, c0:c1] = False
    return grid

def cell_to_metres(cell, cell_metres=1.0):
    return np.array([cell[0], cell[1]], dtype=float) * cell_metres

def save_map(truck_grid, drone_grid, config, path="map.npz"):
    np.savez_compressed(path, truck_grid = truck_grid, drone_grid = drone_grid, config_json = np.array([json.dumps(config)]))
    print(f"Map saved to {path}")


def load_map(path="map.npz"):
    data   = np.load(path, allow_pickle=True)
    config = json.loads(str(data["config_json"][0]))
    return data["truck_grid"], data["drone_grid"], config

def validate_key_points(truck_grid, drone_grid, config):
    for name, cell, grid in [
        ("truck_start", tuple(config["truck_start"]), truck_grid),
        ("truck_end", tuple(config["truck_end"]), truck_grid),
        ("drone_delivery", tuple(config["drone_delivery"]), drone_grid),
    ]:
        col, row = cell
        ok = grid[row, col]
        print(f"{name} = {cell} {'PASSABLE' if ok else 'BLOCKED: change this point'}")
        if not ok:
            raise ValueError(f"{name} cell {cell} is inside a restricted zone: adjust coordinates.")

if __name__ == "__main__":
    print("Step 1: Map Creation")
    gs = map_config["grid_size"]
    print("Building occupancy grids.")
    truck_grid = create_occupancy_grid(map_config["truck_restricted"], gs, TRUCK_MARGIN)
    drone_grid = create_occupancy_grid(map_config["drone_restricted"],  gs, DRONE_MARGIN)
    print("Validating key points.")
    validate_key_points(truck_grid, drone_grid, map_config)
    print("Saving map and visualisation.")
    save_map(truck_grid, drone_grid, map_config, "map.npz")
    visualise_occupancy_grid(truck_grid, drone_grid, map_config, "step1_map.png")
