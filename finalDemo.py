import cv2
import numpy as np
import astar

# Constants for map size (in centimeters)
MAP_X_SIZE = 1654  # map width (in cm)
MAP_Y_SIZE = 821  # map height (in cm)
ROBOT_SIZE = 46  # safe distance around obstacles (in cm)

# Global variables
xbot = None  # Store the position of XBOT as a single tuple
goal = None  # Store the goal position as a single tuple
path = []
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
SEPARATOR = f"{COLOR_BLUE}{'='*50}{COLOR_RESET}"


# Function to log a message with a specific color
def log_message(message, color=COLOR_RESET):
    print(f"{color}{message}{COLOR_RESET}")


# Function to reset the map to its initial state
def reset_map():
    global background_img, img
    img = background_img.copy()
    log_message("Map reset to initial state.", COLOR_YELLOW)

def add():
    if xbot:
        cv2.circle(img, xbot, 2, (0, 255, 0), -1)
        cv2.rectangle(img, (xbot[0] - ROBOT_SIZE, xbot[1] - ROBOT_SIZE),
                      (xbot[0] + ROBOT_SIZE, xbot[1] + ROBOT_SIZE), (0, 255, 0), 1)
    if goal:
        cv2.circle(img, goal, 2, (0, 0, 255), -1)
    for obs in obstacles:
        cv2.circle(img, obs, int(ROBOT_SIZE / 2), (0, 0, 0), -1)


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
                obstacles.append((x, y))
                log_message(f"Obstacle added at position: ({x}, {y})", COLOR_YELLOW)
                cv2.circle(img, (x, y), int(ROBOT_SIZE / 2), (0, 0, 0), -1)


# Function to handle keypress events
def handle_keypress(key):
    global xbot, goal, img, path, obstacles

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

    elif key == ord('p'):  # Plan Path
        reset_map()
        add()
        log_message(SEPARATOR)
        log_message("Planning path...", COLOR_YELLOW)
        if xbot and goal:
            log_message(f"XBOT: {xbot}, Goal: {goal}", COLOR_GREEN)
            unsafe_mask = astar.precompute_safe_zones(obstacles, MAP_X_SIZE, MAP_Y_SIZE, ROBOT_SIZE)
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
        path = []
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
