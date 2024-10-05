import numpy as np


def willObjectCollideWithRobot(P_A0, V_A, S_A, P_B0, V_B, S_B, t_max, d_min, inches_per_point):
    """
    Determines if two moving objects (robot and another object) will collide based on their paths, velocities, and speeds.

    Parameters:
    - P_A0: np.array([x, y]) - Initial position of Robot A (your robot) in inches.
    - V_A: np.array([vx, vy]) - Unit vector for Robot A's velocity (direction).
    - S_A: float - Speed of Robot A in inches per second.
    - P_B0: np.array([x, y]) - Initial position of Robot B (other robot/object) in inches.
    - V_B: np.array([vx, vy]) - Unit vector for Robot B's velocity (direction).
    - S_B: float - Speed of Robot B in inches per second.
    - t_max: float - The maximum time (in seconds) to predict the movement of both objects.
    - d_min: float - Minimum safe distance between the two objects (in inches).
    - inches_per_point: float - The conversion factor for how many inches each point represents in real-world units.

    Returns:
    - bool: True if the objects will collide, False if they won't.
    """

    for t in np.linspace(0, t_max, num=1000):
        P_A_t = P_A0 + V_A * S_A * t
        P_B_t = P_B0 + V_B * S_B * t
        distance = np.linalg.norm(P_A_t - P_B_t) * inches_per_point
        if distance < d_min:
            return True
    return False


P_A0 = np.array([0, 0])  # Initial position of Robot A
V_A = np.array([1, 0])   # Moving to the right
S_A = 10               # Speed of Robot A

P_B0 = np.array([100, 0])  # Initial position of Robot B
V_B = np.array([1, 0])    # Moving to the left
S_B = 10                 # Speed of Robot B

collision = willObjectCollideWithRobot(P_A0, V_A, S_A, P_B0, V_B, S_B, t_max=10, d_min=10, inches_per_point=1)
print(collision)

