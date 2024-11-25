import random
from queue import PriorityQueue

import numpy as np

# Positions are in Y, X not X, Y
# Define constants
INCHES_PER_SQUARE = 5  # Accuracy - Inches per square
TOTAL_MAP_SIZE_X_INCHES = 651  # 2024 FRC Game Field Size X
TOTAL_MAP_SIZE_Y_INCHES = 323  # 2024 FRC Game Field Size Y
ROBOT_LOCATION = (2, 1)  # Robot location
GOAL_LOCATION = (10, 30)  # Goal location

# Calculate the total size of the grid in terms of squares
map_size_x_squares = TOTAL_MAP_SIZE_X_INCHES // INCHES_PER_SQUARE
map_size_y_squares = TOTAL_MAP_SIZE_Y_INCHES // INCHES_PER_SQUARE

# Generate random obstacles
obstacles = [(random.randint(0, map_size_y_squares - 1), random.randint(0, map_size_x_squares - 1)) for _ in range(190)]


def remove_obstacles(obstacles, positions):
    return [obstacle for obstacle in obstacles if obstacle not in positions]


# Remove obstacles at the robot and goal locations
obstacles = remove_obstacles(obstacles, [ROBOT_LOCATION, GOAL_LOCATION])


def willObjectCollideWithRobot(P_A0, V_A, P_B0, V_B, t_max, d_min, inches_per_point):
    for t in np.linspace(0, t_max, num=1000):
        P_A_t = (P_A0[0] + V_A[0] * t, P_A0[1] + V_A[1] * t)
        P_B_t = (P_B0[0] + V_B[0] * t, P_B0[1] + V_B[1] * t)
        distance = np.linalg.norm(np.array(P_A_t) - np.array(P_B_t)) * inches_per_point
        if distance < d_min:
            collision_position = ((P_A_t[0] + P_B_t[0]) / 2, (P_A_t[1] + P_B_t[1]) / 2)
            return True, t, collision_position, distance
    return False, None, None, None


def mark_collision_zone_on_grid(grid, P_B_predicted, d_min, inches_per_point):
    radius = int(d_min / inches_per_point)
    x_center, y_center = int(P_B_predicted[0]), int(P_B_predicted[1])

    for x in range(x_center - radius, x_center + radius + 1):
        for y in range(y_center - radius, y_center + radius + 1):
            if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
                    grid[x][y] = 1  # Mark the cell as an obstacle


def heuristic(a, b, D=1, D2=1.414):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)


def a_star_search(start, goal, obstacles, map_size_x, map_size_y):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while not open_set.empty():
        current = open_set.get()[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return reversed path

        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < map_size_y and 0 <= neighbor[1] < map_size_x and neighbor not in obstacles:
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    open_set.put((f_score[neighbor], neighbor))

    return []


def reroute_a_star(start, goal, obstacles, P_B0, V_B, t_max, d_min, inches_per_point):
    P_B_predicted = [P_B0[0] + V_B[0] * t_max, P_B0[1] + V_B[1] * t_max]  # Return as list
    grid = [[0 for _ in range(map_size_x_squares)] for _ in range(map_size_y_squares)]
    mark_collision_zone_on_grid(grid, P_B_predicted, d_min, inches_per_point)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                obstacles.add((i, j))
    return a_star_search(start, goal, obstacles, map_size_x_squares, map_size_y_squares), P_B_predicted


def display_map(size_x, size_y, robot, goal, obstacles, path):
    grid = [['.' for _ in range(size_x)] for _ in range(size_y)]
    grid[robot[0]][robot[1]] = 'R'
    grid[goal[0]][goal[1]] = 'G'
    for obs in obstacles:
        grid[obs[0]][obs[1]] = 'X'
    for step in path:
        if step != robot and step != goal:
            grid[step[0]][step[1]] = '*'

    for row in grid:
        print(' '.join(row))


def log_step(step_number, message):
    print(f"{step_number}. {message}")


def main():
    log_step(1, "Calculating path...")
    path = a_star_search(ROBOT_LOCATION, GOAL_LOCATION, set(obstacles), map_size_x_squares, map_size_y_squares)

    if path:
        log_step(2, "Path found.")
        log_step(3, f"Path: {path}")
    else:
        log_step(2, "No path found.")
        return

    log_step(4, "Calculating for collisions...")

    V_A = (0,10)

    P_B0 = (2, 100)
    V_B = (0, -10)


    collision_detected, collision_time, collision_position, collision_distance = willObjectCollideWithRobot(
        ROBOT_LOCATION, V_A, P_B0,
        V_B, t_max=0.2, d_min=10,
        inches_per_point=INCHES_PER_SQUARE)

    if collision_detected:
        log_step(
            5,
            f"Collision detected at time {collision_time:.2f} seconds! "
            f"Predicted collision position: {collision_position[0]:.2f} inches, {collision_position[1]:.2f} inches. "
            f"Distance at collision: {collision_distance:.2f} inches (threshold: {10} inches)."
        )
        log_step(6, "Rerouting new path...")
        new_path, pb_predicted = reroute_a_star(ROBOT_LOCATION, GOAL_LOCATION, set(obstacles), P_B0, V_B, t_max=1, d_min=10,
                                  inches_per_point=INCHES_PER_SQUARE)
        if new_path:
            log_step(7, f"New Path: {new_path}")
        else:
            log_step(7, "No new path found.")
    else:
        log_step(5, "No collision detected.")
        new_path = path

    display_map(map_size_x_squares, map_size_y_squares, ROBOT_LOCATION, GOAL_LOCATION, obstacles, new_path)


if __name__ == "__main__":
    main()
