
# PROMPT:
# The constraint:  a person is dancing. The  motion length is T. I need to constrain on the velocity magnitude for the three keyframes. The velocity magnitude of the first frame is 2 m/s. The velocity magnitude of the central frame is zero. The velocity magnitude of the last frame is 2 m/s. You can only pick functions provided.


def compute_velocity_magnitude_error(motion, keyframes, target_velocities):
    total_error = 0

    for i, frame_index in enumerate(keyframes):
        # Get the velocity magnitude for the current keyframe
        velocity_magnitude = get_velocity_magnitude(motion, frame_index)

        # Calculate the error based on the difference from the target velocity
        error = MSE(velocity_magnitude, target_velocities[i])

        # Add the error to the total error
        total_error += error

    return total_error

def get_velocity_magnitude(motion, frame_index):
    # Use the provided function to get the velocity vector
    velocity_vector = getVelocity(motion, frame_index)

    # Calculate the velocity magnitude using the provided function
    velocity_magnitude = magnitude(velocity_vector)

    return velocity_magnitude

# Provided functions
def getVelocity(motion, frame_index):
    # Replace this with your logic for getting the velocity vector for the given frame
    # For simplicity, we assume a placeholder function that returns a constant vector
    return (1.0, 0.0, 0.0)

def magnitude(vector):
    # Calculate the magnitude of a 3D vector
    return (vector[0]**2 + vector[1]**2 + vector[2]**2)**0.5

def MSE(a, b):
    # Mean Squared Error between values a and b
    return (a - b) ** 2

# Example usage:
motion_data = [
    {"joint1": (x1, y1, z1), ...},  # Frame 0
    {"joint1": (x2, y2, z2), ...},  # Frame 1
    # ... (more frames)
    {"joint1": (xn, yn, zn), ...},  # Last frame (Frame n)
]

# Specify keyframes and target velocities
keyframes = [0, len(motion_data) // 2, len(motion_data) - 1]
target_velocities = [2.0, 0.0, 2.0]

# Calculate the total error
total_error = compute_velocity_magnitude_error(motion_data, keyframes, target_velocities)
print("Total Velocity Magnitude Error:", total_error)

