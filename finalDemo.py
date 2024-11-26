import cv2
import numpy as np
import astar
import cv2
import numpy as np
import json
import time

# Constants for map size (in centimeters)
MAP_X_SIZE = 1654  # map width (in cm)
MAP_Y_SIZE = 821  # map height (in cm)
ROBOT_SIZE = 26  # safe distance around obstacles (in cm)

# Global variables
xbot = None  # Store the position of XBOT as a single tuple
goal = None  # Store the goal position as a single tuple
#----------
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

background_img = cv2.imread("map.png")
if background_img is not None:
    background_img = cv2.resize(background_img, (MAP_X_SIZE, MAP_Y_SIZE))
else:
    print("\033[91mError: Background image not found!\033[0m")
    background_img = np.ones((MAP_Y_SIZE, MAP_X_SIZE, 3), dtype=np.uint8) * 255

img = background_img.copy()

# ANSI color codes for console logging
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"
SEPARATOR = f"{COLOR_BLUE}{'=' * 50}{COLOR_RESET}"


# Function to log a message with a specific color
def log_message(message, color=COLOR_RESET):
    print(f"{color}{message}{COLOR_RESET}")


# Function to reset the map to its initial state
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


def add(angle=0):
    if xbot:
        # Draw the XBOT as a circle
        cv2.circle(img, xbot, 2, (0, 255, 0), -1)
        if xbot_velocity > 0:
            cv2.putText(img, f"{xbot_velocity:.2f} cm/s",
                        (xbot[0] - 30, xbot[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 2)
        # Get the rotated rectangle points using cv2.boxPoints
        box = cv2.boxPoints(((xbot[0], xbot[1]), (ROBOT_SIZE, ROBOT_SIZE), angle))
        box = np.int32(box)  # Convert float to int for drawing (use np.int0)

        # Draw the rotated rectangle
        cv2.polylines(img, [box], True, (255, 100, 0), 2)  # Green rectangle

    if goal:
        # Draw the goal as a red circle
        cv2.circle(img, goal, 2, (0, 0, 255), -1)

    for obs in obstacles:
        # Draw obstacles as black circles
        cv2.circle(img, obs, 2, (0, 0, 0), -1)
    for opp in opponent_robots:
        cv2.circle(img, opp, 2, (0, 255, 0), -1)

        # Get the rotated rectangle points using cv2.boxPoints
        box = cv2.boxPoints(((opp[0], opp[1]), (ROBOT_SIZE, ROBOT_SIZE), 0))
        box = np.int32(box)
        cv2.polylines(img, [box], True, (255, 255, 0), 2)
    if placing_robot and robot_cursor_position:
        # Draw the XBOT as a circle
        if robot_cursor_velocity > 0:
            cv2.putText(img, f"{robot_cursor_velocity:.2f} cm/s",
                        (robot_cursor_position[0] - 30, robot_cursor_position[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 1)
            print(angle)
            end_point = (int(robot_cursor_position[0] + robot_cursor_velocity * np.cos(np.radians(angle))),
                         int(robot_cursor_position[1] + robot_cursor_velocity * np.sin(np.radians(angle))))

            # Draw the direction arrow for the robot cursor
            cv2.arrowedLine(img, robot_cursor_position, end_point, (255, 0, 0), 2)  # Arrow in blue

        cv2.circle(img, robot_cursor_position, 2, (0, 255, 0), -1)

        # Get the rotated rectangle points using cv2.boxPoints
        box = cv2.boxPoints(((robot_cursor_position[0], robot_cursor_position[1]), (ROBOT_SIZE, ROBOT_SIZE), 0))
        box = np.int32(box)  # Convert float to int for drawing (use np.int0)

        # Draw the rotated rectangle
        cv2.polylines(img, [box], True, (255, 255, 0), 2)


def mouse_movement(event, x, y, flags, param):
    global robot_cursor_position, placing_robot, prev_robot_cursor_position, prev_robot_cursor_time, img, \
        robot_cursor_velocity, robot_cursor_angle

    if placing_robot:
        robot_cursor_position = (x, y)

        # Check if previous position and time exist
        if prev_robot_cursor_position is not None and prev_robot_cursor_time is not None:
            time_diff = time.time() - prev_robot_cursor_time

            distance = np.sqrt((x - prev_robot_cursor_position[0]) ** 2 + (y - prev_robot_cursor_position[1]) ** 2)
            if time_diff != 0:
                robot_cursor_velocity = distance / time_diff
            dx = x - prev_robot_cursor_position[0]
            dy = y - prev_robot_cursor_position[1]
            print(np.degrees(np.arctan2(dy, dx)))
            robot_cursor_angle = np.degrees(np.arctan2(dy, dx))
        # Update the previous position and time after the calculation
        prev_robot_cursor_position = (x, y)
        prev_robot_cursor_time = time.time()


def place_path():
    global path, waypoints
    if path and waypoints:
        for step in path:
            cv2.circle(img, step, 1, (255, 0, 0), -1)
        for point in waypoints:
            cv2.circle(img, point, 3, (0, 255, 255), -1)


# Function to select positions for XBOT, Goal, or Obstacles
def select_position(event, x, y, flags, param):
    global xbot, goal, img, obstacles

    if param == 'X':  # Set XBOT position
        if event == cv2.EVENT_LBUTTONDOWN:
            xbot = (x, y)  # Store XBOT position as a tuple
            log_message(f"XBOT set at position: ({x}, {y})", COLOR_GREEN)
            reset_map()
            add()

    elif param == 'G':  # Set Goal position
        if event == cv2.EVENT_LBUTTONDOWN:
            goal = (x, y)  # Store Goal position as a tuple
            log_message(f"Goal set at position: ({x}, {y})", COLOR_GREEN)
            reset_map()
            add()

    elif param == 'O':  # Add Obstacles
        if event == cv2.EVENT_LBUTTONDOWN or (flags & cv2.EVENT_FLAG_LBUTTON):
            if x < MAP_X_SIZE and y < MAP_Y_SIZE:
                obstacle = (x, y)
                if obstacle not in obstacles:  # Check if the obstacle already exists
                    obstacles.append(obstacle)
                    log_message(f"Obstacle added at position: ({x}, {y})", COLOR_YELLOW)
                    cv2.circle(img, (x, y), 2, (0, 0, 0), -1)
                else:
                    log_message(f"Obstacle at ({x}, {y}) already exists.", COLOR_RED)  # Inform the user if duplicate


# Function to handle keypress events
def handle_keypress(key):
    global xbot, goal, img, path, obstacles, waypoints, placing_robot, robot_cursor_position, opponent_robots,\
        prev_robot_cursor_time, prev_robot_cursor_position, xbot_velocity, prev_xbot_position, prev_xbot_time

    if key == ord('x'):  # Set XBOT position
        log_message(SEPARATOR)
        log_message("Click to set the XBOT position.", COLOR_BLUE)
        log_message(SEPARATOR)
        cv2.setMouseCallback("Path Planning", select_position, 'X')
    elif key == ord('a'):  # Toggle opponent robot placement mode
        placing_robot = not placing_robot  # Toggle the flag
        if placing_robot:
            log_message(SEPARATOR)
            log_message("Opponent robot placement mode ON. Move the cursor.", COLOR_BLUE)
            log_message("Press any other key to place the robot and exit this mode.", COLOR_BLUE)
            log_message(SEPARATOR)
            cv2.setMouseCallback("Path Planning", mouse_movement)  # Track mouse movement
        else:
            if robot_cursor_position:
                opponent_robots.append(robot_cursor_position)  # Place the robot
                log_message(f"Opponent robot placed at position: {robot_cursor_position}", COLOR_GREEN)
                robot_cursor_position = None
            prev_robot_cursor_time = None
            prev_robot_cursor_position = None
            reset_map()
            place_path()
            add()
    elif key == ord('g'):  # Set Goal position
        log_message(SEPARATOR)
        log_message("Click to set the Goal position.", COLOR_BLUE)
        log_message(SEPARATOR)
        cv2.setMouseCallback("Path Planning", select_position, 'G')

    elif key == ord('o'):  # Add Obstacles
        log_message(SEPARATOR)
        log_message("Click to add obstacles.", COLOR_BLUE)
        log_message(SEPARATOR)
        cv2.setMouseCallback("Path Planning", select_position, 'O')
    elif key == ord('f'):  # Start demo
        if path is None or waypoints is None:
            log_message("There is no path.", COLOR_RED)
            return
        log_message(SEPARATOR)
        log_message("Running demo...", COLOR_BLUE)
        log_message(SEPARATOR)
        current = 0

        while current < len(path):
            xbot = path[current]
            if prev_xbot_position is not None and prev_xbot_time is not None:
                # Calculate XBOT velocity based on previous position and time
                time_diff = time.time() - prev_xbot_time
                distance = np.sqrt((xbot[0] - prev_xbot_position[0]) ** 2 + (xbot[1] - prev_xbot_position[1]) ** 2)

                if time_diff > 0:
                    xbot_velocity = distance / time_diff  # cm/s
                else:
                    xbot_velocity = 0

            # Update previous position and time
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

            # Display the velocity
            if xbot_velocity > 0:
                cv2.putText(img, f"{xbot_velocity:.2f} cm/s",
                            (xbot[0] - 30, xbot[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 2)

            # Draw the path and waypoints
            for step in path:
                cv2.circle(img, step, 1, (255, 0, 0), -1)
            for point in waypoints:
                cv2.circle(img, point, 3, (0, 255, 255), -1)

            cv2.imshow("Path Planning", img)
            cv2.waitKey(2)
            current += 1
        xbot_velocity = 0
        log_message("Demo finished.", COLOR_BLUE)
        log_message(SEPARATOR)

    elif key == ord('p'):  # Plan Path

        if xbot and goal:
            if xbot == goal:
                log_message("The XBOT and goal are the same!", COLOR_RED)
                return
            reset_map()
            add()
            log_message(SEPARATOR)
            log_message("Planning path...", COLOR_YELLOW)
            log_message(f"XBOT: {xbot}, Goal: {goal}", COLOR_GREEN)
            unsafe_mask = astar.precompute_safe_zones(static_obstacles + obstacles + opponent_robots, MAP_X_SIZE,
                                                      MAP_Y_SIZE, ROBOT_SIZE)
            path = astar.a_star_search(xbot, goal, unsafe_mask)
            if path is None:
                log_message("No path found!", COLOR_RED)
                return
            waypoints = astar.generate_waypoints(path, 500)
            log_message(f"Path found: {path}", COLOR_GREEN)
            log_message(f"Waypoints extracted: {waypoints}", COLOR_GREEN)

            # Draw the path and waypoints
            for step in path:
                cv2.circle(img, step, 1, (255, 0, 0), -1)  # Path in blue
            for point in waypoints:
                cv2.circle(img, point, 3, (0, 255, 255), -1)  # Waypoints in yellow

        else:
            log_message("Please set XBOT and Goal positions first!", COLOR_RED)

    elif key == ord('c'):  # Clear the map
        log_message("Clearing the map...", COLOR_YELLOW)
        xbot = None
        goal = None
        path = None
        waypoints = None
        obstacles = []
        opponent_robots = []
        placing_robot = False
        robot_cursor_position = None
        reset_map()


# Main loop to run the demo
def run_demo():
    global robot_cursor_velocity
    # cv2.namedWindow("Path Planning", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Path Planning", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Path Planning", img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        handle_keypress(key)
        if placing_robot:
            reset_map()
            place_path()
            if prev_robot_cursor_position == robot_cursor_position and (
                    time.time() - (prev_robot_cursor_time or 0)) > 0.1:
                robot_cursor_velocity = 0
            add()
        if xbot_velocity == 0:
            reset_map()
            place_path()
            add(0)
        cv2.imshow("Path Planning", img)
        if key == ord('q'):
            log_message("Exiting the program. Goodbye!", COLOR_GREEN)
            break
    cv2.destroyAllWindows()


# Entry point
if __name__ == "__main__":
    load_static_obstacles()
    log_message("Welcome to the Path Planning Demo!", COLOR_BLUE)
    log_message("Instructions:", COLOR_YELLOW)
    log_message("- Press 'x' to set XBOT position", COLOR_YELLOW)
    log_message("- Press 'g' to set Goal position", COLOR_YELLOW)
    log_message("- Press 'o' to add obstacles", COLOR_YELLOW)
    log_message("- Press 'p' to plan the path", COLOR_YELLOW)
    log_message("- Press 'c' to clear the map", COLOR_YELLOW)
    log_message("- Press 'q' to quit", COLOR_YELLOW)
    log_message(SEPARATOR)
    run_demo()
