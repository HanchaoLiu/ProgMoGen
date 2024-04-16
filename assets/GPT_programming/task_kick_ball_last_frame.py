# PROMPT:
# Task: a person is kicking a ball with his left foot in the last frame. The position of the ball is known. Use provide functions.

def compute_kick_error_last_frame(motion, ball_position):
    total_error = 0

    # Constants and thresholds
    max_allowable_distance = 0.2  # Maximum allowable distance between left foot and ball

    # Get the coordinates of the left foot in the last frame
    left_foot_last_frame = motion[-1].get("left_foot", None)

    # Ensure the left foot is present in the last frame
    if left_foot_last_frame is not None:
        # Use provided function to calculate the distance between left foot and ball
        distance_left_foot_to_ball = DistToPoint(left_foot_last_frame, ball_position)

        # Add constraint to total error
        total_error += max(distance_left_foot_to_ball - max_allowable_distance, 0)

        # You may add more constraints based on specific requirements

    return total_error

# Provided function
def DistToPoint(point, target_point):
    # Replace this with your logic for calculating the distance between a point and another point
    # For simplicity, we assume a placeholder function that returns Euclidean distance
    return magnitude(np.array(point) - np.array(target_point))

# Provided function
def magnitude(vector):
    # Calculate the magnitude of a 3D vector
    return (vector[0]**2 + vector[1]**2 + vector[2]**2)**0.5

# Example usage:
ball_position = (x_ball, y_ball, z_ball)  # Replace with the actual coordinates of the ball

motion_data = [
    # ... (motion frames leading up to the last frame)
    {"left_foot": (xn, yn, zn), ...},  # Last frame (Frame n)
]

# Calculate the total error for the last frame
total_error_last_frame = compute_kick_error_last_frame(motion_data, ball_position)
print("Total Kick Error (Last Frame):", total_error_last_frame)

