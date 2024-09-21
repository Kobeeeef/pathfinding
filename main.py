import random
from enum import Enum
from queue import PriorityQueue

# Define variables (replacing constants)
inches_per_square = 10  # Accuracy - Inches Per Square
total_map_size_x_inches = 651  # 2024 FRC Game Field Size X
total_map_size_y_inches = 323  # 2024 FRC Game Field Size Y
robot_location = (2, 1)  # Robot location
goal_location = (10, 30)  # Goal/Note location

# Calculate the total size of the grid in terms of squares
map_size_x_squares = total_map_size_x_inches // inches_per_square
map_size_y_squares = total_map_size_y_inches // inches_per_square

# Generate random obstacles
obstacles = [(random.randint(0, map_size_y_squares - 1), random.randint(0, map_size_x_squares - 1)) for _ in range(15)]
print(obstacles)

def heuristic(a, b):
    """Calculate the Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


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
path = a_star_search(robot_location, goal_location, set(obstacles), map_size_x_squares, map_size_y_squares)
print(path)




# Display the map in the console
def display_map(size_x, size_y, robot, goal, obstacles, path):
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
display_map(map_size_x_squares, map_size_y_squares, robot_location, goal_location, obstacles, path)
