import math
import random
from queue import PriorityQueue

# Define constants
INCHES_PER_SQUARE = 10  # Accuracy - Inches per square
TOTAL_MAP_SIZE_X_INCHES = 651  # 2024 FRC Game Field Size X
TOTAL_MAP_SIZE_Y_INCHES = 323  # 2024 FRC Game Field Size Y
ROBOT_LOCATION = (2, 1)  # Robot location
GOAL_LOCATION = (10, 30)  # Goal location

# Calculate the total size of the grid in terms of squares
map_size_x_squares = TOTAL_MAP_SIZE_X_INCHES // INCHES_PER_SQUARE
map_size_y_squares = TOTAL_MAP_SIZE_Y_INCHES // INCHES_PER_SQUARE

# Generate random obstacles
obstacles = [(random.randint(0, map_size_y_squares - 1), random.randint(0, map_size_x_squares - 1)) for _ in range(100)]
print("Obstacles:", obstacles)


def heuristic(a, b):
    """
    Calculate the heuristic distance between two points using squared Euclidean distance.

    Parameters:
    a (tuple): The coordinates of the first point (x, y).
    b (tuple): The coordinates of the goal point (x, y).
    D (float): The cost factor applied to the squared distance (default is 1).

    Returns:
    float: The heuristic distance based on squared Euclidean distance.
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return math.sqrt((dx * dx + dy * dy))


def a_star_search(start, goal, obstacles, map_size_x, map_size_y):
    """A* search implementation."""
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while not open_set.empty():
        current = open_set.get()[1]

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return reversed path

        # Get neighbors
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < map_size_y and 0 <= neighbor[1] < map_size_x and neighbor not in obstacles:
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    open_set.put((f_score[neighbor], neighbor))

    return []  # Return an empty path if no path is found


# Run the A* algorithm
path = a_star_search(ROBOT_LOCATION, GOAL_LOCATION, set(obstacles), map_size_x_squares, map_size_y_squares)
print("Path:", path)


def display_map(size_x, size_y, robot, goal, obstacles, path):
    """Display the map in the console."""
    grid = [['.' for _ in range(size_x)] for _ in range(size_y)]
    grid[robot[0]][robot[1]] = 'R'  # Mark the robot location
    grid[goal[0]][goal[1]] = 'G'  # Mark the goal location
    for obs in obstacles:
        grid[obs[0]][obs[1]] = 'X'  # Mark obstacles
    for step in path:
        if step != robot and step != goal:
            grid[step[0]][step[1]] = '*'  # Mark the path

    # Print the grid
    for row in grid:
        print(' '.join(row))


# Display the map with the path
display_map(map_size_x_squares, map_size_y_squares, ROBOT_LOCATION, GOAL_LOCATION, obstacles, path)
