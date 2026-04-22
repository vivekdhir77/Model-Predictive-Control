import math
import heapq
import random
import numpy as np
from map import load_map
from config import T_STEP, TRUCK_SPEED, V_CRUISE, SAMPLE_S, W_BAR_SEARCH


def A_star_path(start, goal, grid):
    rows, cols = grid.shape
    sc, sr = start
    gc, gr = goal
    if not (0 <= sc < cols and 0 <= sr < rows):
        return None
    if not (0 <= gc < cols and 0 <= gr < rows):
        return None
    if not grid[sr, sc] or not grid[gr, gc]:
        return None
    if start == goal:
        return [start]
    def h(cell):
        c, r = cell
        return math.hypot(c - gc, r - gr)
    open_set = []
    heapq.heappush(open_set, (h(start), 0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    neighbours = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
    while open_set:
        _, g, current = heapq.heappop(open_set)
        if g > g_score.get(current, float("inf")):
            continue
        if current == goal:
            path, node = [], current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            return path[::-1]
        cx, cy = current
        for dx, dy, cost in neighbours:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            if not grid[ny, nx]:
                continue
            new_g = g + cost
            nb = (nx, ny)
            if new_g < g_score.get(nb, float("inf")):
                g_score[nb] = new_g
                came_from[nb] = current
                heapq.heappush(open_set, (new_g + h(nb), new_g, nb))
    return None

def segment_collision_free(n1, n2, grid):
    rows, cols = grid.shape
    x1, y1 = n1
    x2, y2 = n2
    steps = max(abs(x2 - x1), abs(y2 - y1), 1)
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x1 + t * (x2 - x1)))
        y = int(round(y1 + t * (y2 - y1)))
        if not (0 <= x < cols and 0 <= y < rows):
            return False
        if not grid[y, x]:
            return False
    return True


def rrt_smooth(path, grid, passes=3):
    for _ in range(passes):
        i = 0
        smoothed = [path[0]]
        while i < len(path)-1:
            j = len(path)-1
            while j > i + 1:
                if segment_collision_free(path[i], path[j], grid):
                    break
                j-=1
            smoothed.append(path[j])
            i = j
        path =smoothed
    return path


def standard_rrt(start, goal, grid, step_size=20, max_iter=4000, goal_bias=0.10, goal_tol=15):
    """Standard RRT with no wind bias — grows purely toward random samples."""
    rows, cols = grid.shape
    sc, sr = start
    gc, gr = goal
    if not (0 <= sc < cols and 0 <= sr < rows): return None
    if not (0 <= gc < cols and 0 <= gr < rows): return None
    if not grid[sr, sc] or not grid[gr, gc]:    return None
    if start == goal: return [start]

    parent = {start: None}

    for _ in range(max_iter):
        if random.random() < goal_bias:
            rx, ry = gc, gr
        else:
            rx =random.randint(0, cols - 1)
            ry =random.randint(0, rows - 1)

        nearest = min(parent.keys(), key=lambda n: math.hypot(n[0]-rx, n[1]-ry))
        nx, ny = nearest
        dx, dy = rx - nx, ry - ny
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            continue
        new_x = int(round(nx + step_size*dx/dist))
        new_y = int(round(ny + step_size*dy/dist))
        new_x = max(0, min(new_x, cols - 1))
        new_y = max(0, min(new_y, rows - 1))
        new_node = (new_x, new_y)
        if new_node in parent or not grid[new_y, new_x]:
            continue
        if not segment_collision_free(nearest, new_node, grid):
            continue
        parent[new_node] = nearest
        if math.hypot(new_x-gc, new_y-gr) <= goal_tol:
            if segment_collision_free(new_node, goal, grid):
                parent[goal] = new_node
                break

    if goal not in parent:
        return None
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def wind_biased_rrt(start, goal, grid, wind=None, step_size=20, max_iter=4000, goal_bias=0.10, goal_tol=15):
    if wind is None:
        wind = W_BAR_SEARCH

    rows, cols = grid.shape
    sc, sr = start
    gc, gr = goal
    if not (0 <= sc < cols and 0 <= sr < rows): return None
    if not (0 <= gc < cols and 0 <= gr < rows): return None
    if not grid[sr, sc] or not grid[gr, gc]:    return None
    if start == goal: 
        return [start]
    wind_mag = math.hypot(wind[0], wind[1])
    wind_hat = np.array(wind) / wind_mag if wind_mag > 1e-9 else np.zeros(2)
    wind_offset = int(round(step_size * 0.5))
    parent = {start: None}

    for _ in range(max_iter):
        if random.random() < goal_bias:
            rx, ry = gc, gr
        else:
            rx = random.randint(0, cols - 1)
            ry = random.randint(0, rows - 1)
        if random.random() < 0.10:
            rx = int(np.clip(rx + wind_offset * wind_hat[0], 0, cols - 1))
            ry = int(np.clip(ry + wind_offset * wind_hat[1], 0, rows - 1))

        nearest = min(parent.keys(), key=lambda n: math.hypot(n[0]-rx, n[1]-ry))
        nx, ny = nearest
        dx, dy = rx - nx, ry - ny
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            continue
        new_x = int(round(nx + step_size * dx / dist))
        new_y = int(round(ny + step_size * dy / dist))
        new_x = max(0, min(new_x, cols - 1))
        new_y = max(0, min(new_y, rows - 1))
        new_node = (new_x, new_y)
        if new_node in parent or not grid[new_y, new_x]:
            continue
        if not segment_collision_free(nearest, new_node, grid):
            continue
        parent[new_node] = nearest
        if math.hypot(new_x - gc, new_y - gr) <= goal_tol:
            if segment_collision_free(new_node, goal, grid):
                parent[goal] = new_node
                break

    if goal not in parent:
        return None
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def gamma(path, speed=None):
    arc = {0: 0.0}
    for i in range(1, len(path)):
        dc = path[i][0]-path[i-1][0]
        dr = path[i][1]-path[i-1][1]
        arc[i] = arc[i-1] + math.hypot(dc, dr)
    return arc

def estimate_truck_position(arc, truck_path, distance):
    l, r = 0, len(arc) - 1
    while l <= r:
        mid = (l + r)//2
        if arc[mid] < distance:
            l = mid + 1
        else:
            r = mid - 1
    idx = min(l, len(truck_path) - 1)
    return truck_path[idx]

def launch_search(drone_grid, config, truck_path):
    K_L_star = 0
    truck_arc = gamma(truck_path)
    distance_star = float("inf")
    drone_path_1_star = None
    drone_path_2_star = None
    truck_estimated_position_return_star = None
    path_length_star_1 = float("inf")
    path_length_star_2 = float("inf")
    stats = []
    for i in range(0, len(truck_path), SAMPLE_S):
        col, row = truck_path[i]
        if not drone_grid[row, col]:
            continue
        drone_path_1 = A_star_path(truck_path[i], tuple(config["drone_delivery"]), drone_grid)
        if drone_path_1 is None:
            continue
        drone_arc_1 = gamma(drone_path_1)
        drone_path_length_1 = drone_arc_1[len(drone_path_1) - 1]
        fly_time = drone_path_length_1/V_CRUISE
        truck_estimated_position_delivery = estimate_truck_position(truck_arc, truck_path, truck_arc[i]+(fly_time*TRUCK_SPEED))
        for return_estimate_time in np.arange(1.0, 3.0, 0.5):
            truck_estimated_position_return = estimate_truck_position(truck_arc, truck_path, truck_arc[i]+(return_estimate_time*fly_time+fly_time)*TRUCK_SPEED)
            drone_path_2 = A_star_path(tuple(config["drone_delivery"]), truck_estimated_position_return, drone_grid)
            if drone_path_2 is None:
                continue
            drone_arc_2 = gamma(drone_path_2)
            drone_path_length_2 = drone_arc_2[len(drone_path_2) - 1]
            stats.append({
                "path_length_1": drone_path_length_1,
                "path_length_2": drone_path_length_2,
                "truck_launch_time": i,
                "truck_return_time": i+(return_estimate_time*fly_time)+fly_time,
            })
            if drone_path_length_1 + drone_path_length_2 < distance_star:
                distance_star=drone_path_length_1 + drone_path_length_2
                K_L_star = i
                drone_path_1_star = drone_path_1
                drone_path_2_star = drone_path_2
                truck_estimated_position_return_star = truck_estimated_position_return
                path_length_star_1 = drone_path_length_1
                path_length_star_2 = drone_path_length_2
    return K_L_star, path_length_star_1, path_length_star_2, drone_path_1_star, drone_path_2_star, truck_estimated_position_return_star, stats


if __name__ == "__main__":
    truck_grid, drone_grid, config = load_map("map.npz")
    truck_path = A_star_path(tuple(config["truck_start"]), tuple(config["truck_end"]), truck_grid)
    if truck_path is None:
        raise RuntimeError("Truck A* failed. Check truck_start/truck_end in map configuration.")
    K_L_star, path_length_star_1, path_length_star_2, drone_path_1_star, drone_path_2_star, truck_estimated_position_return_star, stats = launch_search(drone_grid, config, truck_path)
    print(f"K_L_star = {K_L_star}")
    print(f"Path length 1 A*: {path_length_star_1}")
    print(f"Path length 2 A*: {path_length_star_2}")

    print(f"Truck estimated position return: {truck_estimated_position_return_star}")
    if drone_path_1_star is None:
        print("No A* path found from truck to delivery point")
    if drone_path_2_star is None:
        print("No A* path found from delivery point to truck return point")

    drone_path_1_rrt = wind_biased_rrt(truck_path[K_L_star], tuple(config["drone_delivery"]), drone_grid, wind=W_BAR_SEARCH)
    drone_path_2_rrt = wind_biased_rrt(tuple(config["drone_delivery"]), truck_estimated_position_return_star, drone_grid, wind=W_BAR_SEARCH)

    drone_path_1_std_rrt = standard_rrt(truck_path[K_L_star], tuple(config["drone_delivery"]), drone_grid)
    drone_path_2_std_rrt = standard_rrt(tuple(config["drone_delivery"]), truck_estimated_position_return_star, drone_grid)

    if drone_path_1_rrt is None:
        print("No wind-biased RRT path found from truck to delivery point")
    if drone_path_2_rrt is None:
        print("No wind-biased RRT path found from delivery point to truck return point")
    if drone_path_1_std_rrt is None:
        print("No standard RRT path found from truck to delivery point")
    if drone_path_2_std_rrt is None:
        print("No standard RRT path found from delivery point to truck return point")

    path_length_rrt_1 = gamma(drone_path_1_rrt)[len(drone_path_1_rrt) - 1] if drone_path_1_rrt else float("inf")
    path_length_rrt_2 = gamma(drone_path_2_rrt)[len(drone_path_2_rrt) - 1] if drone_path_2_rrt else float("inf")
    path_length_std_rrt_1 = gamma(drone_path_1_std_rrt)[len(drone_path_1_std_rrt) - 1] if drone_path_1_std_rrt else float("inf")
    path_length_std_rrt_2 = gamma(drone_path_2_std_rrt)[len(drone_path_2_std_rrt) - 1] if drone_path_2_std_rrt else float("inf")

    print(f"Path length wind-biased RRT 1 : {path_length_rrt_1}")
    print(f"Path length wind-biased RRT 2 : {path_length_rrt_2}")
    print(f"Path length standard RRT 1 : {path_length_std_rrt_1}")
    print(f"Path length standard RRT 2 : {path_length_std_rrt_2}")

    np.savez("init_path_results.npz",
        truck_path=np.array(truck_path, dtype=object),
        K_L_star=np.array(K_L_star),
        path_length_star_1=np.array(path_length_star_1),
        path_length_star_2=np.array(path_length_star_2),
        drone_path_1_star=np.array(drone_path_1_star, dtype=object),
        drone_path_2_star=np.array(drone_path_2_star, dtype=object),
        truck_estimated_position_return_star=np.array(truck_estimated_position_return_star, dtype=object),
        stats=np.array(stats, dtype=object),
        drone_path_1_rrt=np.array(drone_path_1_rrt, dtype=object),
        drone_path_2_rrt=np.array(drone_path_2_rrt, dtype=object),
        drone_path_1_std_rrt=np.array(drone_path_1_std_rrt, dtype=object),
        drone_path_2_std_rrt=np.array(drone_path_2_std_rrt, dtype=object),
    )
    print("Saved results to init_path_results.npz")
    