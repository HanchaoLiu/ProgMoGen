
# PROMPT:
# Task: a person is balancing on one leg, and his right foot is on a wooden stake.  Consider the geometric and physical constraints for the task and write the code

def compute_error(motion):
    total_error = 0

    # Constants and thresholds
    desired_height = 1.0  # Desired height for the center of mass
    stake_position = (x_stake, y_stake, z_stake)  # Coordinates of the wooden stake
    max_allowable_tilt = 15  # Maximum allowable tilt angle for the body

    # Iterate through frames
    for frame_index, joints in enumerate(motion):
        # Get the right foot and center of mass coordinates for the current frame
        right_foot = joints.get("right_foot", None)
        center_of_mass = joints.get("center_of_mass", None)

        if right_foot is not None and center_of_mass is not None:
            # Geometric constraint: Distance from right foot to wooden stake
            distance_right_foot_to_stake = DistToPoint(right_foot, stake_position)

            # Geometric constraint: Distance between center of mass and right foot
            distance_com_to_right_foot = DistToPoint(center_of_mass, right_foot)

            # Physical constraint: Control center of mass height
            height_error = MSE(center_of_mass[2], desired_height)

            # Physical constraint: Limit body tilt angle
            tilt_angle = compute_tilt_angle(center_of_mass, right_foot)
            tilt_error = max(tilt_angle - max_allowable_tilt, 0)

            # Combine individual geometric and physical constraints
            total_error += max(distance_right_foot_to_stake - stake_radius, 0)  # Assuming stake_radius is defined
            total_error += max(distance_com_to_right_foot - max_allowable_distance, 0)  # Assuming max_allowable_distance is defined
            total_error += height_error
            total_error += tilt_error

            # You may add more constraints based on specific requirements

    return total_error

def compute_tilt_angle(center_of_mass, reference_point):
    # Compute the tilt angle (in degrees) between the vertical and the line connecting the center of mass and the reference point
    delta_x = center_of_mass[0] - reference_point[0]
    delta_y = center_of_mass[1] - reference_point[1]
    tilt_angle = math.degrees(math.atan2(delta_y, delta_x))
    return tilt_angle

# Example usage:
motion_data = [
    {"right_foot": (x1, y1, z1), "center_of_mass": (cx1, cy1, cz1), ...},  # Frame 0
    {"right_foot": (x2, y2, z2), "center_of_mass": (cx2, cy2, cz2), ...},  # Frame 1
    # ... (more frames)
    {"right_foot": (xn, yn, zn), "center_of_mass": (cxn, cyn, czn), ...},  # Last frame (Frame n)
]

stake_radius = 0.05  # Set your stake radius accordingly
max_allowable_distance = 0.3  # Set your maximum allowable distance accordingly

total_error = compute_error(motion_data)
print("Total Error:", total_error)

