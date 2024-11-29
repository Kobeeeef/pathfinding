import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt
import math
import matplotlib.pyplot as plt
import cv2

# Map dimensions
MAP_X_SIZE = 1654  # Map width (in cm)
MAP_Y_SIZE = 821  # Map height (in cm)

# Heuristic weight
PRIORITIZE_GOAL_WEIGHT = 30

# Safe zone parameters
SAFE_DISTANCE = 10  # Distance to maintain from obstacles (in cm)
UNSAFE_AREA_PENALTY = 50  # Penalty for unsafe zones
DYNAMIC_OBSTACLE_PENALTY = 1.2  # Multiplier for dynamic obstacle cost

# Visualization settings
NODE_DRAW_COLOR = (0, 255, 0)  # Color for nodes being visualized
NODE_DRAW_RADIUS = 1  # Radius for drawing nodes

# Path generation parameters
INFLECTION_ANGLE_THRESHOLD = math.pi / 6  # Minimum angle to detect inflection points (radians)

# Load background image
background_img = cv2.imread("map.png")
if background_img is not None:
    background_img = cv2.resize(background_img, (MAP_X_SIZE, MAP_Y_SIZE))
else:
    print("\033[91mError: Background image not found!\033[0m")
    background_img = np.ones((MAP_Y_SIZE, MAP_X_SIZE, 3), dtype=np.uint8) * 255

img = background_img.copy()


def heuristic(a, b):
    """
    Calculates the heuristic for A* using a weighted Manhattan distance.
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return PRIORITIZE_GOAL_WEIGHT * (dx + dy) + (1.414 - 2) * min(dx, dy)


def precompute_safe_zones(obstacles, dynamic_obstacles, map_size_x, map_size_y, safe_distance):
    """
    Precomputes the unsafe zones and their associated costs.
    """
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


def a_star_search(start, goal, unsafe_mask, cost_mask, dynamic_obstacles):
    """
    A* search algorithm with additional penalties for unsafe zones and dynamic obstacles.
    """
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

    node_expansion_count = 0
    closed_set = set()

    while open_set:
        current = heapq.heappop(open_set)[1]
        node_expansion_count += 1
        cv2.circle(img, current, NODE_DRAW_RADIUS, NODE_DRAW_COLOR, -1)
        if current in closed_set:
            continue
        closed_set.add(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            # cv2.imshow("ASTAR LOOKUP", img)
            # cv2.waitKey(1)
            print(f"Nodes expanded: {node_expansion_count}")
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
                    move_cost *= DYNAMIC_OBSTACLE_PENALTY

                # Add cost from cost mask
                move_cost += cost_mask[neighbor[1], neighbor[0]]

                # Penalize unsafe areas instead of skipping
                move_cost += UNSAFE_AREA_PENALTY if unsafe_mask[neighbor[1], neighbor[0]] else 0

                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None


def generate_waypoints(path, waypoint_interval):
    """
    Generates waypoints along the path at regular intervals or at inflection points.
    """
    if not path or len(path) < 2:
        raise ValueError("Path must contain at least two points (start and goal).")

    waypoints = [path[0]]  # Always include the start point
    accumulated_distance = 0

    for i in range(1, len(path) - 1):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        accumulated_distance += distance

        prev_dx = path[i][0] - path[i - 1][0]
        prev_dy = path[i][1] - path[i - 1][1]
        curr_dx = path[i + 1][0] - path[i][0]
        curr_dy = path[i + 1][1] - path[i][1]

        prev_slope = math.atan2(prev_dy, prev_dx)
        curr_slope = math.atan2(curr_dy, curr_dx)

        # Detect inflection points
        if abs(curr_slope - prev_slope) > INFLECTION_ANGLE_THRESHOLD:
            waypoints.append(path[i])
            accumulated_distance = 0

        # Add waypoints at regular intervals
        if accumulated_distance >= waypoint_interval:
            waypoints.append(path[i])
            accumulated_distance = 0

    # Always include the final point
    if path[-1] not in waypoints:
        waypoints.append(path[-1])

    return waypoints


def debug_visualize_masks(unsafe_mask, cost_mask):
    """
    Visualizes the unsafe and cost masks for debugging purposes.
    """
    plt.title("Unsafe Mask")
    plt.imshow(unsafe_mask, cmap='gray')
    plt.colorbar()
    plt.show()

    plt.title("Cost Mask")
    plt.imshow(cost_mask, cmap='hot')
    plt.colorbar()
    plt.show()


def calculate_dynamic_window(xbot, velocity, robot_size, map_x_size, map_y_size):
    """
    Dynamically calculates and draws a square around the robot that adjusts to fit within the map boundaries.

    Args:
        xbot: Tuple (x, y) representing the robot's current position.
        velocity: The robot's current velocity.
        robot_size: Size of the robot (radius or half-size for the bounding square).
        map_x_size: Width of the map.
        map_y_size: Height of the map.

    Returns:
        The points of the dynamically adjusted square around the robot.
    """
    # Define the maximum side length of the square
    max_side_length = (robot_size * 2) + velocity / 2

    # Calculate the half side length
    half_side = max_side_length / 2

    # Determine boundaries to prevent the square from going out of the map
    left = max(xbot[0] - half_side, 0)
    right = min(xbot[0] + half_side, map_x_size)
    top = max(xbot[1] - half_side, 0)
    bottom = min(xbot[1] + half_side, map_y_size)

    # Adjust the square dimensions if close to boundaries
    adjusted_width = right - left
    adjusted_height = bottom - top

    # Create the square as a list of points
    square_points = [
        (int(left), int(top)),
        (int(right), int(top)),
        (int(right), int(bottom)),
        (int(left), int(bottom))
    ]

    return np.array(square_points, dtype=np.int32)


def predict_collisions(opponents, path, safe_distance):
    """
    Predicts potential collisions between the robot path and opponents' robots.

    Args:
        opponents: List of tuples, each containing:
            - opponent_position: Tuple (x, y) representing the current position of an opponent robot.
            - opponent_velocity: Scalar representing the velocity of the opponent robot.
            - opponent_angle: Angle in radians representing the direction of the opponent robot's movement.
        path: List of (x, y) points representing the planned path of the robot.
        safe_distance: Minimum distance to maintain between the robot path and the opponent robots.

    Returns:
        Tuple:
            - List of path indices where potential collisions are predicted.
            - List of predicted opponent positions where collisions are likely to occur.
    """
    conflicts = []  # To store the indices of the path points where collisions are predicted
    collision_positions = []  # To store the positions of the opponent robots where collisions are predicted

    # Loop through each opponent
    for opponent_position, opponent_velocity, opponent_angle in opponents:
        if not opponent_angle or not opponent_position:
            continue
        # Predict the opponent's next position based on their velocity and direction
        if opponent_velocity > 0:
            predicted_opponent_position = (
                int(opponent_position[0] + opponent_velocity * np.cos(opponent_angle)),
                int(opponent_position[1] + opponent_velocity * np.sin(opponent_angle))
            )
        else:
            # For stationary robots, use their current position
            predicted_opponent_position = opponent_position

        # Check each path point for potential collision
        for i, point in enumerate(path):
            # Calculate the distance from the opponent position to the path point
            distance_to_opponent = np.linalg.norm(np.array(point) - np.array(predicted_opponent_position))
            if distance_to_opponent <= safe_distance:
                conflicts.append(i)  # Append the path index where collision is detected
                collision_positions.append(predicted_opponent_position)  # Append the opponent's predicted position

    # Remove duplicates in the conflicts list if needed
    conflicts = list(set(conflicts))

    return conflicts, collision_positions
