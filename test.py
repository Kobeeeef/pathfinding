import cv2
import numpy as np

import astar

# Constants for map size (in centimeters)
MAP_X_SIZE = 1200  # map width (in cm)
MAP_Y_SIZE = 650  # map height (in cm)
ROBOT_SIZE = 30  # safe distance around obstacles (in cm),
# Global variables
xbot = None  # Store the position of XBOT as a single tuple
goal = None  # Store the goal position as a single tuple

# Create the map image
img = np.ones((MAP_Y_SIZE, MAP_X_SIZE, 3), dtype=np.uint8) * 255  # White background

path = []
obstacles = []


# Function to select positions for XBOT or Goal
def select_position(event, x, y, flags, param):
    global xbot, goal, img, obstacles

    if param == 'X':  # Set XBOT position
        if event == cv2.EVENT_LBUTTONDOWN:
            xbot = (x, y)  # Store XBOT position as a tuple
            print(f"XBOT set at position: ({x}, {y})")
            img = np.ones((MAP_Y_SIZE, MAP_X_SIZE, 3), dtype=np.uint8) * 255  # Clear the map
            cv2.circle(img, xbot, 2, (0, 255, 0), -1)
            cv2.rectangle(img, (x - ROBOT_SIZE, y - ROBOT_SIZE), (x + ROBOT_SIZE, y + ROBOT_SIZE), (0, 255, 0), 1)
            if goal:  # If a goal is set, draw it
                cv2.circle(img, goal, 2, (0, 0, 255), -1)
            for obs in obstacles:  # Redraw obstacles
                cv2.circle(img, obs, 1, (0, 0, 0), -1)  # Draw each obstacle as black dots
    elif param == 'G':  # Set Goal position
        if event == cv2.EVENT_LBUTTONDOWN:
            goal = (x, y)  # Store Goal position as a tuple
            print(f"Goal set at position: ({x}, {y})")
            img = np.ones((MAP_Y_SIZE, MAP_X_SIZE, 3), dtype=np.uint8) * 255
            if xbot:
                cv2.circle(img, xbot, 2, (0, 255, 0), -1)
                cv2.rectangle(img, (xbot[0] - ROBOT_SIZE, xbot[1] - ROBOT_SIZE), (xbot[0] + ROBOT_SIZE, xbot[1] + ROBOT_SIZE), (0, 255, 0), 1)
            cv2.circle(img, goal, 2, (0, 0, 255), -1)
            for obs in obstacles:
                cv2.circle(img, obs, 1, (0, 0, 0), -1)
    elif param == 'O':  # Add Obstacles
        if event == cv2.EVENT_LBUTTONDOWN or (flags & cv2.EVENT_FLAG_LBUTTON):  # On mouse down or drag
            if x < MAP_X_SIZE and y < MAP_Y_SIZE:
                obstacles.append((x, y))
                print(f"Obstacle added at position: ({x}, {y})")
                cv2.circle(img, (x, y), 1, (0, 0, 0), -1)  # Draw the obstacle as a small black dot


# Function to handle keypress events and mouse callbacks
def handle_keypress(key):
    global xbot, goal, img, path, obstacles
    if key == ord('x'):  # Set XBOT position
        print("Click to add XBOT position")
        cv2.setMouseCallback("Path Planning", select_position, 'X')
    elif key == ord('g'):  # Set Goal position
        print("Click to add Goal position")
        cv2.setMouseCallback("Path Planning", select_position, 'G')
    elif key == ord('o'):  # Set Goal position
        print("Click to add Obstacle position")
        cv2.setMouseCallback("Path Planning", select_position, 'O')
    elif key == ord('p'):
        print("Planning path...")
        img = np.ones((MAP_Y_SIZE, MAP_X_SIZE, 3), dtype=np.uint8) * 255
        if xbot:
            cv2.circle(img, xbot, 2, (0, 255, 0), -1)
            cv2.rectangle(img, (xbot[0] - ROBOT_SIZE, xbot[1] - ROBOT_SIZE), (xbot[0] + ROBOT_SIZE, xbot[1] + ROBOT_SIZE), (0, 255, 0), 1)
        cv2.circle(img, goal, 2, (0, 0, 255), -1)
        for obs in obstacles:
            cv2.circle(img, obs, 1, (0, 0, 0), -1)
        if xbot and goal:
            print(f"XBOT: {xbot}, Goal: {goal}")
            unsafe_mask = astar.precompute_safe_zones(obstacles, MAP_X_SIZE, MAP_Y_SIZE, ROBOT_SIZE)

            path = astar.a_star_search(xbot, goal, unsafe_mask)
            if path is None:
                print("No path found!")
                return
            for step in path:
                cv2.circle(img, step, 1, (255, 0, 0), -1)

            print(f"Path found: {path}")
        else:
            print("Please set XBOT and Goal positions first!")
    elif key == ord('c'):
        print("Clearing the map...")
        xbot = None
        goal = None
        path = []
        obstacles = []
        img = np.ones((MAP_Y_SIZE, MAP_X_SIZE, 3), dtype=np.uint8) * 255  # Reset to white background


# Main function to run the demo
def run_demo():
    global img
    cv2.imshow("Path Planning", img)

    while True:
        key = cv2.waitKey(1) & 0xFF

        handle_keypress(key)

        cv2.imshow("Path Planning", img)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()
