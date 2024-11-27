import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt
import math
import matplotlib.pyplot as plt
import cv2

# Map dimensions
MAP_X_SIZE = 1654  # map width (in cm)
MAP_Y_SIZE = 821  # map height (in cm)

PRIORITIZE_GOAL_WEIGHT = 30


def heuristic(a, b):
    return PRIORITIZE_GOAL_WEIGHT * (abs(a[0] - b[0]) + abs(a[1] - b[1]))


# Precompute unsafe zones and cost mask
def precompute_safe_zones(obstacles, dynamic_obstacles, map_size_x, map_size_y, safe_distance):
    grid = np.zeros((map_size_y, map_size_x), dtype=bool)

    # Mark static obstacles
    for obs in obstacles:
        if 0 <= obs[0] < map_size_x and 0 <= obs[1] < map_size_y:
            grid[obs[1], obs[0]] = True

    # Mark dynamic obstacles
    for dyn_obs in dynamic_obstacles:
        if 0 <= dyn_obs[0] < map_size_x and 0 <= dyn_obs[1] < map_size_y:
            grid[dyn_obs[1], dyn_obs[0]] = True

    # Compute distance transform for gradient cost
    distance_map = distance_transform_edt(~grid)
    cost_mask = np.clip(safe_distance - distance_map, 0, safe_distance)
    unsafe_mask = cost_mask > 0

    return unsafe_mask, cost_mask


# Load background map
background_img = cv2.imread("map.png")
if background_img is not None:
    background_img = cv2.resize(background_img, (MAP_X_SIZE, MAP_Y_SIZE))
else:
    print("\033[91mError: Background image not found!\033[0m")
    background_img = np.ones((MAP_Y_SIZE, MAP_X_SIZE, 3), dtype=np.uint8) * 255

img = background_img.copy()


# A* search algorithm
def a_star_search(start, goal, unsafe_mask, cost_mask, dynamic_obstacles):
    img = background_img.copy()
    map_size_y, map_size_x = unsafe_mask.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    dynamic_set = set(dynamic_obstacles)

    # Ensure goal is reachable
    if 0 <= goal[0] < map_size_x and 0 <= goal[1] < map_size_y:
        unsafe_mask[goal[1], goal[0]] = False
        cost_mask[goal[1], goal[0]] = 0

    i = 0
    while open_set:
        current = heapq.heappop(open_set)[1]
        i += 1
        cv2.circle(img, current, 1, (0, 255, 0), -1)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            cv2.imshow("ASTAR LOOKUP", img)
            cv2.waitKey(1)
            print(f"Nodes expanded: {i}")
            return path[::-1]

        neighbors = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < map_size_x and 0 <= neighbor[1] < map_size_y:
                move_cost = 1.414 if dx != 0 and dy != 0 else 1

                # Dynamic obstacle cost penalty
                if neighbor in dynamic_set:
                    move_cost *= 1.2  # Adjusted dynamic obstacle penalty

                # Add cost from cost mask
                move_cost += cost_mask[neighbor[1], neighbor[0]]

                # Penalize unsafe areas instead of skipping
                move_cost += 50 if unsafe_mask[neighbor[1], neighbor[0]] else 0

                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None


# Generate waypoints every nth cm along the path
def generate_waypoints(path, n):
    if not path or len(path) < 2:
        raise ValueError("Path must contain at least two points (start and goal).")

    waypoints = [path[0]]  # Always include the start point
    accumulated_distance = 0

    for i in range(1, len(path) - 1):
        # Calculate Euclidean distance between consecutive points
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        accumulated_distance += distance

        # Check for inflection points by comparing slopes of consecutive segments
        prev_dx = path[i][0] - path[i - 1][0]
        prev_dy = path[i][1] - path[i - 1][1]
        curr_dx = path[i + 1][0] - path[i][0]
        curr_dy = path[i + 1][1] - path[i][1]

        # Calculate slopes
        prev_slope = math.atan2(prev_dy, prev_dx)
        curr_slope = math.atan2(curr_dy, curr_dx)

        # Detect inflection points
        if abs(curr_slope - prev_slope) > math.pi / 6:  # Threshold for a sharp turn
            waypoints.append(path[i])
            accumulated_distance = 0  # Reset accumulated distance after inflection

        # Add waypoints at regular intervals
        if accumulated_distance >= n:
            waypoints.append(path[i])
            accumulated_distance = 0

    # Always include the final point
    if path[-1] not in waypoints:
        waypoints.append(path[-1])

    return waypoints


# Debugging and Visualization
def debug_visualize_masks(unsafe_mask, cost_mask):
    plt.title("Unsafe Mask")
    plt.imshow(unsafe_mask, cmap='gray')
    plt.colorbar()
    plt.show()

    plt.title("Cost Mask")
    plt.imshow(cost_mask, cmap='hot')
    plt.colorbar()
    plt.show()
