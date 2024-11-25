import heapq

import numpy as np
from scipy.ndimage import distance_transform_edt


def heuristic(a, b):
    D = 1
    D2 = 1.414
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)


def precompute_safe_zones(obstacles, map_size_x, map_size_y, safe_distance):
    grid = np.zeros((map_size_y, map_size_x), dtype=bool)
    for obs in obstacles:
        grid[obs[1], obs[0]] = True
    distance_map = distance_transform_edt(~grid)
    unsafe_mask = distance_map <= safe_distance
    return unsafe_mask


def a_star_search(start, goal, unsafe_mask):
    map_size_y, map_size_x = unsafe_mask.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        neighbors = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < map_size_x and 0 <= neighbor[1] < map_size_y and not unsafe_mask[
                neighbor[1], neighbor[0]]:
                move_cost = 1.414 if dx != 0 and dy != 0 else 1
                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None
