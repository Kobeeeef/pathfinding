import json
import time

import cv2
import numpy as np
import copy
import astar

# Constants for map size (in centimeters)
MAP_X_SIZE = 1654  # map width (in cm)
MAP_Y_SIZE = 821  # map height (in cm)
ROBOT_SIZE = 80  # safe distance around obstacles (in cm) MINIMUM 15
SMOOTH_ANGLE_STEP_SIZE = 1.3
xbot = None
goal = None

path = None
waypoints = None
static_obstacles = []
obstacles = []
prev_xbot_position = None
prev_xbot_time = None
xbot_velocity = 0
robot_cursor_position = None
prev_robot_cursor_time = None
prev_robot_cursor_position = None
robot_cursor_angle = None
robot_cursor_velocity = 0
placing_robot = False
opponent_robots = []
demo_running = False
background_img = cv2.imread("map.png")
if background_img is not None:
    background_img = cv2.resize(background_img, (MAP_X_SIZE, MAP_Y_SIZE))
else:
    print("\033[91mError: Background image not found!\033[0m")
    background_img = np.ones((MAP_Y_SIZE, MAP_X_SIZE, 3), dtype=np.uint8) * 255

img = background_img.copy()

COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"
SEPARATOR = f"{COLOR_BLUE}{'=' * 50}{COLOR_RESET}"
remove_path_visuals = True
full_screen = False

unsafe_mask, cost_mask = (None, None)

BLUE_SCORING = (1523, 556)
RED_SCORING = (127, 556)


def log_message(message, color=COLOR_RESET):
    print(f"{color}{message}{COLOR_RESET}")


def reset_map():
    global background_img, img
    img = background_img.copy()


def load_static_obstacles(filename="obstacles.json"):
    global static_obstacles
    try:
        with open(filename, "r") as f:
            static_obstacles = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode the file {filename}.")


# ---------- SMOOTH ANGLE TRANSITION ----------
current_angle = 0


def smooth_angle_update(target_angle, current_angle, step_size=SMOOTH_ANGLE_STEP_SIZE):
    if abs(target_angle - current_angle) <= step_size:
        return target_angle
    elif target_angle > current_angle:
        return current_angle + step_size
    else:
        return current_angle - step_size


# ---------- SMOOTH ANGLE TRANSITION ----------

def add(angle=0):
    global current_angle

    # Smoothly update the current angle towards the desired angle
    current_angle = smooth_angle_update(angle, current_angle)

    if xbot:
        cv2.circle(img, xbot, 2, (0, 255, 0), -1)
        if xbot_velocity > 0:
            cv2.putText(img, f"{xbot_velocity:.2f} cm/s",
                        (xbot[0] - 40, xbot[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2)

        box = cv2.boxPoints(((xbot[0], xbot[1]), (ROBOT_SIZE, ROBOT_SIZE), current_angle))
        box = np.int32(box)

        cv2.polylines(img, [box], True, (255, 100, 0), 2)

    if goal:
        cv2.circle(img, goal, 2, (0, 0, 255), -1)

    for obs in obstacles:
        cv2.circle(img, obs, 2, (0, 0, 0), -1)
    for opp in opponent_robots:
        cv2.circle(img, opp, 2, (0, 255, 0), -1)

        box = cv2.boxPoints(((opp[0], opp[1]), (ROBOT_SIZE, ROBOT_SIZE), 0))
        box = np.int32(box)
        cv2.polylines(img, [box], True, (255, 255, 0), 2)
    if placing_robot and robot_cursor_position:

        if robot_cursor_velocity > 0:
            cv2.putText(img, f"{robot_cursor_velocity:.2f} cm/s",
                        (robot_cursor_position[0] - 30, robot_cursor_position[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 2)

            end_point = (int(robot_cursor_position[0] + 80 * np.cos(np.radians(robot_cursor_angle))),
                         int(robot_cursor_position[1] + 80 * np.sin(np.radians(robot_cursor_angle))))

            cv2.arrowedLine(img, robot_cursor_position, end_point, (0, 0, 0), 2)

        cv2.circle(img, robot_cursor_position, 2, (0, 255, 0), -1)

        box = cv2.boxPoints(((robot_cursor_position[0], robot_cursor_position[1]), (ROBOT_SIZE, ROBOT_SIZE), 0))
        box = np.int32(box)

        cv2.polylines(img, [box], True, (255, 255, 0), 2)


def mouse_movement(event, x, y, flags, param):
    global robot_cursor_position, placing_robot, prev_robot_cursor_position, prev_robot_cursor_time, img, \
        robot_cursor_velocity, robot_cursor_angle

    if placing_robot:
        robot_cursor_position = (x, y)

        if prev_robot_cursor_position is not None and prev_robot_cursor_time is not None:
            time_diff = time.time() - prev_robot_cursor_time
            if time_diff < 0.1:
                return
            distance = np.sqrt((x - prev_robot_cursor_position[0]) ** 2 + (y - prev_robot_cursor_position[1]) ** 2)
            if time_diff != 0:
                robot_cursor_velocity = distance / time_diff

            dx = x - prev_robot_cursor_position[0]
            dy = y - prev_robot_cursor_position[1]

            robot_cursor_angle = np.degrees(np.arctan2(dy, dx))

        prev_robot_cursor_position = (x, y)
        prev_robot_cursor_time = time.time()


def place_path():
    global path, waypoints
    if not remove_path_visuals:
        if path and waypoints:
            for step in path:
                cv2.circle(img, step, 1, (255, 0, 0), -1)
            for point in waypoints:
                cv2.circle(img, point, 4, (0, 255, 255), 2)





# Function to adjust the path to avoid obstacles



def reconstruct_full_path(local_path, window_min_x, window_min_y):
    full_path = []
    for point in local_path:
        global_x = point[0] + window_min_x
        global_y = point[1] + window_min_y
        full_path.append((global_x, global_y))
    return full_path


def select_position(event, x, y, flags, param):
    global xbot, goal, img, obstacles

    if param == 'X':
        if event == cv2.EVENT_LBUTTONDOWN:
            xbot = (x, y)
            log_message(f"XBOT set at position: ({x}, {y})", COLOR_GREEN)
            reset_map()
            add()

    elif param == 'G':
        if event == cv2.EVENT_LBUTTONDOWN:
            goal = (x, y)
            log_message(f"Goal set at position: ({x}, {y})", COLOR_GREEN)
            reset_map()
            add()
    elif param == 'GP':
        if event == cv2.EVENT_LBUTTONDOWN:
            goal = (x, y)
            log_message(f"Goal set at position: ({x}, {y})", COLOR_GREEN)
            reset_map()
            add()
            path_plan()

    elif param == 'O':
        if event == cv2.EVENT_LBUTTONDOWN or (flags & cv2.EVENT_FLAG_LBUTTON):
            if x < MAP_X_SIZE and y < MAP_Y_SIZE:
                obstacle = (x, y)
                if obstacle not in obstacles:
                    obstacles.append(obstacle)
                    log_message(f"Obstacle added at position: ({x}, {y})", COLOR_YELLOW)
                    cv2.circle(img, (x, y), 2, (0, 0, 0), -1)
                else:
                    log_message(f"Obstacle at ({x}, {y}) already exists.", COLOR_RED)


def path_plan():
    global path, waypoints
    if xbot and goal:
        if xbot == goal:
            log_message("The XBOT and goal are the same!", COLOR_RED)
            return
        reset_map()
        add()
        log_message(SEPARATOR)
        log_message("Planning path...", COLOR_YELLOW)
        log_message(f"XBOT: {xbot}, Goal: {goal}", COLOR_GREEN)
        # time_before = time.time()
        #
        # log_message(f"Precompute Time: {(time.time() - time_before):.2f}s", COLOR_GREEN)
        # astar.debug_visualize_masks(unsafe_mask, cost_mask)

        time_before = time.time()
        path = astar.a_star_search(xbot, goal, unsafe_mask, cost_mask, obstacles + opponent_robots)
        if path is None:
            log_message("No path found!", COLOR_RED)
            return
        log_message(f"A* Time: {(time.time() - time_before):.2f}s", COLOR_GREEN)
        waypoints = astar.generate_waypoints(path, 800)
        log_message(f"Path: {path}", COLOR_GREEN)
        log_message(f"Waypoints: {waypoints}", COLOR_GREEN)
        if not remove_path_visuals:
            for step in path:
                cv2.circle(img, step, 1, (255, 0, 0), -1)
            for point in waypoints:
                cv2.circle(img, point, 4, (0, 255, 255), 2)

    else:
        log_message("Please set XBOT and Goal positions first!", COLOR_RED)


def handle_keypress(key):
    global xbot, remove_path_visuals, full_screen, demo_running, robot_cursor_velocity, goal, img, path, obstacles, waypoints, placing_robot, robot_cursor_position, opponent_robots, \
        prev_robot_cursor_time, prev_robot_cursor_position, xbot_velocity, prev_xbot_position, prev_xbot_time

    if key == ord('x'):
        log_message(SEPARATOR)
        log_message("Click to set the XBOT position.", COLOR_BLUE)
        log_message(SEPARATOR)
        cv2.setMouseCallback("Path Planning", select_position, 'X')
    elif key == ord('a'):
        placing_robot = not placing_robot
        if placing_robot:
            log_message(SEPARATOR)
            log_message("Opponent robot placement mode ON. Move the cursor.", COLOR_BLUE)
            log_message("Press any other key to place the robot and exit this mode.", COLOR_BLUE)
            log_message(SEPARATOR)
            cv2.setMouseCallback("Path Planning", mouse_movement)
        else:
            if robot_cursor_position:
                opponent_robots.append(robot_cursor_position)
                log_message(f"Opponent robot placed at position: {robot_cursor_position}", COLOR_GREEN)
                robot_cursor_position = None
            prev_robot_cursor_time = None
            prev_robot_cursor_position = None
            reset_map()
            place_path()
            add()
    elif key == ord('g'):
        log_message(SEPARATOR)
        log_message("Click to set the Goal position.", COLOR_BLUE)
        log_message(SEPARATOR)
        cv2.setMouseCallback("Path Planning", select_position, 'G')
    elif key == ord('t'):
        remove_path_visuals = not remove_path_visuals
        log_message(SEPARATOR)
        log_message(
            "The path and waypoints are no longer shown." if remove_path_visuals else
            "The path and waypoints are now shown.",
            COLOR_BLUE)
        log_message(SEPARATOR)
    elif key == ord('k'):
        log_message(SEPARATOR)
        log_message("Click to set the Goal position then Path Plan.", COLOR_BLUE)
        log_message(SEPARATOR)
        cv2.setMouseCallback("Path Planning", select_position, 'GP')
    elif key == ord('r'):
        if demo_running:
            goal = RED_SCORING
            log_message(SEPARATOR)
            log_message("Returning to red speaker...", COLOR_BLUE)
            log_message(SEPARATOR)
            path_plan()
        else:
            log_message("The demo is not running.", COLOR_RED)
    elif key == ord('b'):
        if demo_running:
            goal = BLUE_SCORING
            log_message(SEPARATOR)
            log_message("Returning to blue speaker...", COLOR_BLUE)
            log_message(SEPARATOR)
            path_plan()
        else:
            log_message("The demo is not running.", COLOR_RED)
    elif key == ord('q'):
        log_message("Exiting the program. Goodbye!", COLOR_RED)
        exit(0)
    elif key == ord('s'):
        if full_screen:
            cv2.destroyWindow("Path Planning")
            cv2.imshow("Path Planning", img)
            full_screen = False
        else:
            cv2.destroyWindow("Path Planning")
            cv2.namedWindow("Path Planning", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Path Planning", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Path Planning", img)
            full_screen = True

    elif key == ord('o'):
        log_message(SEPARATOR)
        log_message("Click to add obstacles.", COLOR_BLUE)
        log_message(SEPARATOR)
        cv2.setMouseCallback("Path Planning", select_position, 'O')
    elif key == ord('f'):
        if path is None or waypoints is None:
            log_message("There is no path.", COLOR_RED)
            return

        log_message(SEPARATOR)
        log_message("Running demo with dynamic warping...", COLOR_BLUE)
        log_message(SEPARATOR)

        current = 0
        demo_running = True
        flag = False
        copy_goal = copy.copy(goal)

        while demo_running:
            key = cv2.waitKey(1)

            if key == ord('f'):
                demo_running = False
            elif handle_keypress(key):
                reset_map()
                add(0)
                place_path()
                cv2.imshow("Path Planning", img)

            # Dynamic path update check
            if copy_goal == goal and not flag:
                if current >= len(path):
                    flag = True
                    continue
            else:
                if not path:
                    demo_running = False
                    continue
                if goal != copy_goal:
                    current = 0
                    flag = False
                    copy_goal = copy.copy(goal)
                    continue
                else:
                    xbot_velocity = 0
                    reset_map()
                    add(0)
                    place_path()
                    cv2.imshow("Path Planning", img)
                    continue

            # Current robot position
            xbot = path[current]

            # Predict dynamic obstacle conflicts and warp path
            if prev_xbot_position is not None and prev_xbot_time is not None:
                time_diff = time.time() - prev_xbot_time
                distance = np.sqrt((xbot[0] - prev_xbot_position[0]) ** 2 + (xbot[1] - prev_xbot_position[1]) ** 2)

                if time_diff > 0:
                    xbot_velocity = distance / time_diff
                else:
                    xbot_velocity = 0

            prev_xbot_position = xbot
            prev_xbot_time = time.time()

            # Calculate angle to next point
            if current < len(path) - 1:
                next_point = path[current + 1]
                dx = next_point[0] - xbot[0]
                dy = next_point[1] - xbot[1]
                angle = np.degrees(np.arctan2(dy, dx))
            else:
                angle = 0

            reset_map()
            add(angle)
            predicted_path_obstacles, predicted_path_obstacle_positions = astar.predict_collisions(
                [(robot_cursor_position, robot_cursor_velocity, robot_cursor_angle)], path, ROBOT_SIZE * 2.2)

            # Calculate dynamic window for robot movement
            points = astar.calculate_dynamic_window(
                xbot=xbot,
                velocity=xbot_velocity,
                robot_size=ROBOT_SIZE * 2.5,
                map_x_size=MAP_X_SIZE,
                map_y_size=MAP_Y_SIZE
            )
            cv2.polylines(img, [points], True, (0, 50, 255), 2)

            for i in predicted_path_obstacle_positions:
                cv2.circle(img, i, 20, (0, 0, 255), -1)

            last_path_point = None
            affected_points = []
            for i in range(len(path)):
                path_point = path[i]

                result = cv2.pointPolygonTest(np.array(points), path_point, False)

                if result >= 0:
                    if i >= current:
                        cv2.circle(img, path_point, 1, (0, 255, 0), -1)
                        affected_points.append(path_point)
                        last_path_point = path_point
            predicted_set = set([path[i] for i in predicted_path_obstacles])
            affected_set = set(affected_points)

            if (len(predicted_path_obstacles) > 0 and len(predicted_path_obstacle_positions) > 0 and
                    predicted_set & affected_set):

                unsafe_mask_temp, cost_mask_temp = astar.precompute_safe_zones(static_obstacles, obstacles + opponent_robots + predicted_path_obstacle_positions + [robot_cursor_position],
                                                                     MAP_X_SIZE,
                                                                     MAP_Y_SIZE, ROBOT_SIZE)
                new_path = astar.a_star_search(xbot, last_path_point, unsafe_mask_temp, cost_mask_temp,
                                               predicted_path_obstacle_positions + [robot_cursor_position])
                for i in new_path:
                    cv2.circle(img, i, 5, (20, 255, 255), -1)

                if new_path:
                    # Find the index of xbot position in the original path
                    xbot_index = path.index(xbot)  # Assuming xbot is part of the path

                    # Find the index of the last valid point in the original path
                    last_valid_index = path.index(last_path_point)

                    # Erase the part of the path from xbot to last_valid_point and replace it with new_path
                    updated_path = path[:xbot_index] + new_path[1:] + path[last_valid_index + 1:]

                    # Update the path with the combined new path
                    path = updated_path

                    # Visualize the updated path
                    for i in path:
                        cv2.circle(img, i, 5, (20, 0, 255), -1)

            # Display velocity
            if xbot_velocity > 0:
                cv2.putText(img, f"{xbot_velocity:.2f} cm/s",
                            (xbot[0] - 40, xbot[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 2)

            # Draw waypoints and path
            if not remove_path_visuals:
                for step in path:
                    cv2.circle(img, step, 1, (255, 0, 0), -1)  # Red path points
                for point in waypoints:
                    cv2.circle(img, point, 4, (0, 255, 255), 2)

            cv2.imshow("Path Planning", img)
            current += 1

        xbot_velocity = 0
        log_message("Demo finished.", COLOR_BLUE)
        log_message(SEPARATOR)



    elif key == ord('p'):
        path_plan()

    elif key == ord('c'):
        log_message("Clearing the map...", COLOR_YELLOW)
        demo_running = False
        xbot = None
        goal = None
        path = None
        waypoints = None
        obstacles = []
        opponent_robots = []
        placing_robot = False
        robot_cursor_position = None
        reset_map()
    else:
        return False
    return True


def run_demo():
    global robot_cursor_velocity, unsafe_mask, cost_mask
    time_before = time.time()
    unsafe_mask, cost_mask = astar.precompute_safe_zones(static_obstacles, obstacles + opponent_robots,
                                                         MAP_X_SIZE,
                                                         MAP_Y_SIZE, ROBOT_SIZE)
    log_message(f"Precompute Time: {(time.time() - time_before):.2f}s", COLOR_GREEN)
    cv2.imshow("Path Planning", img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        handle_keypress(key)
        if placing_robot:
            reset_map()
            place_path()
            if (
                    time.time() - (prev_robot_cursor_time or 0)) > 0.15:
                robot_cursor_velocity = 0
            add()
        if xbot_velocity == 0:
            reset_map()
            place_path()
            add(0)
        cv2.imshow("Path Planning", img)
        if key == ord('q'):
            log_message("Exiting the program. Goodbye!", COLOR_RED)
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    load_static_obstacles()
    log_message(SEPARATOR)
    log_message("Welcome to the Path Planning Demo by Kobe!", COLOR_BLUE)
    log_message("Instructions:", COLOR_YELLOW)
    log_message("- Press 'x' to set XBOT position", COLOR_YELLOW)
    log_message("- Press 'g' to set Goal position", COLOR_YELLOW)
    log_message("- Press 'k' to set Goal position & Plan", COLOR_YELLOW)
    log_message("- Press 'o' to add obstacles", COLOR_YELLOW)
    log_message("- Press 'p' to plan the path", COLOR_YELLOW)
    log_message("- Press 'a' to add or move robots", COLOR_YELLOW)
    log_message("- Press 'f' to play the demo", COLOR_YELLOW)
    log_message("- Press 't' to toggle path & waypoint visuals", COLOR_YELLOW)
    log_message("- Press 's' to toggle fullscreen", COLOR_YELLOW)
    log_message("- Press 'c' to clear the map", COLOR_YELLOW)
    log_message("- Press 'b' to go to blue speaker", COLOR_YELLOW)
    log_message("- Press 'r' to go to red speaker", COLOR_YELLOW)
    log_message("- Press 'q' to quit", COLOR_YELLOW)
    log_message(SEPARATOR)
    run_demo()
