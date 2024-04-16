
# PROMPT:
# Task: a person is twisting for two circles. Use the provided functions.


def compute_twisting_error(motion):
    total_error = 0

    # Constants and thresholds
    max_allowable_rotation_diff = 30  # Maximum allowable difference (in degrees) between rotations

    # Iterate through frames
    for frame_index, joints in enumerate(motion):
        # Get the coordinates of relevant joints for the current frame
        hip = joints.get("hip", None)
        left_shoulder = joints.get("left_shoulder", None)
        right_shoulder = joints.get("right_shoulder", None)

        # Ensure all relevant joints are present
        if hip is not None and left_shoulder is not None and right_shoulder is not None:
            # Use provided functions to get the direction vectors of the shoulders
            left_shoulder_direction = getDirection(hip, left_shoulder)
            right_shoulder_direction = getDirection(hip, right_shoulder)

            # Calculate the angles between the shoulder directions and the vertical direction
            angle_diff_left_shoulder = calculate_angle_difference(left_shoulder_direction, (0, 1, 0))
            angle_diff_right_shoulder = calculate_angle_difference(right_shoulder_direction, (0, 1, 0))

            # Add constraints to total error
            total_error += max(angle_diff_left_shoulder - max_allowable_rotation_diff, 0)
            total_error += max(angle_diff_right_shoulder - max_allowable_rotation_diff, 0)

            # You may add more constraints based on specific requirements

    # Add motion dynamics constraints, e.g., rotational velocity constraints
    total_error += compute_rotational_velocity_constraint(motion)

    return total_error

def compute_rotational_velocity_constraint(motion):
    # Placeholder function for computing rotational velocity constraints
    total_rotational_velocity_error = 0

    # Constants and thresholds for rotational velocity constraints
    max_allowable_rotational_velocity = 30  # Maximum allowable rotational velocity (in degrees per frame)

    # Iterate through frames (excluding the first frame)
    for frame_index in range(1, len(motion)):
        # Get the rotational velocity of the shoulders for the current frame
        rotational_velocity_left_shoulder = getRotationalVelocity(
            motion[frame_index]["left_shoulder"], motion[frame_index - 1]["left_shoulder"]
        )
        rotational_velocity_right_shoulder = getRotationalVelocity(
            motion[frame_index]["right_shoulder"], motion[frame_index - 1]["right_shoulder"]
        )

        # Add constraint to total rotational velocity error
        total_rotational_velocity_error += max(
            np.linalg.norm(rotational_velocity_left_shoulder) - max_allowable_rotational_velocity, 0
        )
        total_rotational_velocity_error += max(
            np.linalg.norm(rotational_velocity_right_shoulder) - max_allowable_rotational_velocity, 0
        )

        # You may add more rotational velocity constraints based on specific requirements

    return total_rotational_velocity_error

# Provided function
def getDirection(joint1, joint2):
    # Replace this with your logic for getting the direction vector between two joints
    # For simplicity, we assume a placeholder function that returns a constant vector
    return (1.0, 0.0, 0.0)

# Provided function
def calculate_angle_difference(vector1, vector2):
    # Replace this with your logic for calculating the angle difference between two vectors
    # For simplicity, we assume a placeholder function that returns the angle in degrees
    return angle_between_vectors(vector1, vector2)

# Placeholder function for demonstration purposes
def angle_between_vectors(vector1, vector2):
    # Assume vectors are normalized for simplicity
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Provided function
def getRotationalVelocity(current_direction, previous_direction):
    # Replace this with your logic for calculating the rotational velocity between two directions
    return (np.array(current_direction) - np.array(previous_direction))

# Example usage:
motion_data_twisting = [
    {"hip": (x1, y1, z1), "left_shoulder": (x2, y2, z2), "right_shoulder": (x3, y3, z3), ...},  # Frame 0
    {"hip": (x4, y4, z4), "left_shoulder": (x5, y5, z5), "right_shoulder": (x6, y6, z6), ...},  # Frame 1
    # ... (more frames)
    {"hip": (xm, ym, zm), "left_shoulder": (xn, yn, zn), "right_shoulder": (xo, yo, zo), ...},  # Last frame (Frame n)
]

# Calculate the total error for twisting for two circles
total_error_twisting = compute_twisting_error(motion_data_twisting)
print("Total Twisting Error:", total_error_twisting)

