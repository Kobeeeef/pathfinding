import json
import time

import cv2
import numpy as np

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


def collision_detected():
    global xbot, robot_cursor_position, robot_cursor_velocity, robot_cursor_angle
    if robot_cursor_position and robot_cursor_velocity > 0:

        future_x = robot_cursor_position[0] + robot_cursor_velocity * np.cos(np.radians(robot_cursor_angle))
        future_y = robot_cursor_position[1] + robot_cursor_velocity * np.sin(np.radians(robot_cursor_angle))

        distance = np.sqrt((xbot[0] - future_x) ** 2 + (xbot[1] - future_y) ** 2)
        if distance < (ROBOT_SIZE * 2):
            return True
    return False


# Function to adjust the path to avoid obstacles
def adjust_path_around_obstacles(path):
    global xbot, robot_cursor_position, robot_cursor_velocity, robot_cursor_angle, obstacles

    # Check for collision and adjust the path if necessary
    if collision_detected():
        print("Collision detected! Adjusting path...")

        adjusted_path = []

        # For each segment in the path, check if it's close to an obstacle
        for i in range(len(path) - 1):
            current_point = path[i]
            next_point = path[i + 1]

            # Check if the path segment is close to an obstacle
            for obs in obstacles:
                obs_distance = np.sqrt((obs[0] - current_point[0]) ** 2 + (obs[1] - current_point[1]) ** 2)
                if obs_distance < (ROBOT_SIZE * 3):  # If an obstacle is close to the path
                    # Apply an offset to avoid the obstacle
                    offset_angle = np.radians(45)  # Angle to steer around the obstacle
                    offset_x = int(np.cos(offset_angle) * 20)  # Steer by 20 pixels
                    offset_y = int(np.sin(offset_angle) * 20)

                    # Add a new point to adjust the path around the obstacle
                    adjusted_path.append((current_point[0] + offset_x, current_point[1] + offset_y))
                    adjusted_path.append((next_point[0] + offset_x, next_point[1] + offset_y))

                    # Skip the original path segment, as it's now adjusted
                    break
            else:
                # If no obstacle was found, continue with the normal path segment
                adjusted_path.append(current_point)

        return adjusted_path  # Return the adjusted path

    return path


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
        log_message("Running demo...", COLOR_BLUE)
        log_message(SEPARATOR)

        current = 0
        demo_running = True
        flag = False
        copy_path = path.copy()
        while demo_running:
            key = cv2.waitKey(1)

            if key == ord('f'):
                demo_running = False
            elif handle_keypress(key):
                reset_map()
                add(0)
                place_path()
                cv2.imshow("Path Planning", img)
                cv2.waitKey(1)
            if copy_path == path and not flag:
                if current >= len(path):
                    flag = True
                    continue
            else:
                if not path:
                    demo_running = False
                    continue
                if path != copy_path:
                    current = 0
                    flag = False
                    copy_path = path.copy()
                    continue
                else:
                    xbot_velocity = 0
                    continue

            xbot = path[current]

            if placing_robot:
                reset_map()
                place_path()
                if prev_robot_cursor_position == robot_cursor_position and (
                        time.time() - (prev_robot_cursor_time or 0)) > 0.3:
                    robot_cursor_velocity = 0
            if prev_xbot_position is not None and prev_xbot_time is not None:
                time_diff = time.time() - prev_xbot_time
                distance = np.sqrt((xbot[0] - prev_xbot_position[0]) ** 2 + (xbot[1] - prev_xbot_position[1]) ** 2)

                if time_diff > 0:
                    xbot_velocity = distance / time_diff
                else:
                    xbot_velocity = 0

            prev_xbot_position = xbot
            prev_xbot_time = time.time()

            if current < len(path) - 1:
                next_point = path[current + 1]
                dx = next_point[0] - xbot[0]
                dy = next_point[1] - xbot[1]
                angle = np.degrees(np.arctan2(dy, dx))
            else:
                angle = 0

            reset_map()
            add(angle)

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
            if prev_robot_cursor_position == robot_cursor_position and (
                    time.time() - (prev_robot_cursor_time or 0)) > 0.3:
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
