import cv2
import numpy as np
import astar
import cv2
import numpy as np
import json

# Constants for map size (in centimeters)
MAP_X_SIZE = 1654  # map width (in cm)
MAP_Y_SIZE = 821  # map height (in cm)
ROBOT_SIZE = 26  # safe distance around obstacles (in cm)

# Global variables
xbot = None  # Store the position of XBOT as a single tuple
goal = None  # Store the goal position as a single tuple
path = None
waypoints = None
static_obstacles = []
obstacles = []

# Load and resize the map image
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

        # Get the rotated rectangle points using cv2.boxPoints
        box = cv2.boxPoints(((xbot[0], xbot[1]), (ROBOT_SIZE, ROBOT_SIZE), angle))
        box = np.int32(box)  # Convert float to int for drawing (use np.int0)

        # Draw the rotated rectangle
        cv2.polylines(img, [box], True, (0, 255, 0), 1)  # Green rectangle

    if goal:
        # Draw the goal as a red circle
        cv2.circle(img, goal, 2, (0, 0, 255), -1)

    for obs in obstacles:
        # Draw obstacles as black circles
        cv2.circle(img, obs, 2, (0, 0, 0), -1)


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
    global xbot, goal, img, path, obstacles, waypoints

    if key == ord('x'):  # Set XBOT position
        log_message(SEPARATOR)
        log_message("Click to set the XBOT position.", COLOR_BLUE)
        log_message(SEPARATOR)
        cv2.setMouseCallback("Path Planning", select_position, 'X')

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
            if current < len(path) - 1:
                next_point = path[current + 1]
                dx = next_point[0] - xbot[0]
                dy = next_point[1] - xbot[1]
                angle = np.degrees(np.arctan2(dy, dx))
            else:
                angle = 0
            reset_map()
            add(angle)

            for step in path:
                cv2.circle(img, step, 1, (255, 0, 0), -1)
            for point in waypoints:
                cv2.circle(img, point, 3, (0, 255, 255), -1)
            cv2.imshow("Path Planning", img)
            cv2.waitKey(1)
            current += 1
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
            unsafe_mask = astar.precompute_safe_zones(static_obstacles + obstacles, MAP_X_SIZE, MAP_Y_SIZE, ROBOT_SIZE)
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
        reset_map()


# Main loop to run the demo
def run_demo():
    cv2.imshow("Path Planning", img)
    while True:
        key = cv2.waitKey(1) & 0xFF
        handle_keypress(key)
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
