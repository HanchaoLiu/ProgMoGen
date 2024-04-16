
# PROMPT:
# Task: a person is carrying a large and very heavy ball. The radius of the ball is 0.4 meters. Consider geometric and physical constraints and write the code for the error function.

import math
def compute_error(motion):
    total_error = 0

    # Constants and thresholds
    ball_radius = 0.4  # Radius of the ball
    max_allowable_tilt = 15  # Maximum allowable tilt angle for the body
    max_allowable_distance = 0.6  # Maximum allowable distance from the body to the ball

    # Iterate through frames
    for frame_index, joints in enumerate(motion):
        # Get the hands and center of mass coordinates for the current frame
        left_hand = joints.get("left_hand", None)
        right_hand = joints.get("right_hand", None)
        center_of_mass = joints.get("center_of_mass", None)

        if left_hand is not None and right_hand is not None and center_of_mass is not None:
            # Geometric constraint: Distance between each hand and the ball
            distance_left_hand_to_ball = DistToPoint(left_hand, (x_ball, y_ball, z_ball))
            distance_right_hand_to_ball = DistToPoint(right_hand, (x_ball, y_ball, z_ball))

            # Geometric constraint: Distance between center of mass and the ball
            distance_com_to_ball = DistToPoint(center_of_mass, (x_ball, y_ball, z_ball))

            # Physical constraint: Control center of mass height (assuming the ball is carried at chest level)
            height_error = MSE(center_of_mass[2], chest_level_height)

            # Physical constraint: Limit body tilt angle
            tilt_angle = compute_tilt_angle(center_of_mass, left_hand, right_hand)
            tilt_error = max(tilt_angle - max_allowable_tilt, 0)

            # Combine individual geometric and physical constraints
            total_error += max(distance_left_hand_to_ball - ball_radius, 0)
            total_error += max(distance_right_hand_to_ball - ball_radius, 0)
            total_error += max(distance_com_to_ball - max_allowable_distance, 0)
            total_error += height_error
            total_error += tilt_error

            # You may add more constraints based on specific requirements

    return total_error

def compute_tilt_angle(center_of_mass, left_hand, right_hand):
    # Compute the tilt angle (in degrees) based on the orientation of the hands and the center of mass
    vector_left_hand_to_com = (center_of_mass[0] - left_hand[0], center_of_mass[1] - left_hand[1])
    vector_right_hand_to_com = (center_of_mass[0] - right_hand[0], center_of_mass[1] - right_hand[1])
    tilt_angle = math.degrees(math.acos((vector_left_hand_to_com[0]*vector_right_hand_to_com[0] +
                                          vector_left_hand_to_com[1]*vector_right_hand_to_com[1]) /
                                         (math.sqrt(vector_left_hand_to_com[0]**2 + vector_left_hand_to_com[1]**2) *
                                          math.sqrt(vector_right_hand_to_com[0]**2 + vector_right_hand_to_com[1]**2))))
    return tilt_angle

# Example usage:
motion_data = [
    {"left_hand": (x1, y1, z1), "right_hand": (x2, y2, z2), "center_of_mass": (cx1, cy1, cz1), ...},  # Frame 0
    {"left_hand": (x3, y3, z3), "right_hand": (x4, y4, z4), "center_of_mass": (cx2, cy2, cz2), ...},  # Frame 1
    # ... (more frames)
    {"left_hand": (xn, yn, zn), "right_hand": (xm, ym, zm), "center_of_mass": (cxn, cyn, czn), ...},  # Last frame (Frame n)
]

x_ball, y_ball, z_ball = 0, 0, 0  # Set the position of the ball
chest_level_height = 1.2  # Set the desired chest level height

total_error = compute_error(motion_data)
print("Total Error:", total_error)

