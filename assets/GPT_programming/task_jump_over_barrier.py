
# PROMPT:
# Task: a person jumps over a barrier in the middle frame. The height of the barrier is h0. Use provided functions.



def compute_jump_error(motion, barrier_height):
    total_error = 0

    # Constants and thresholds
    max_allowable_height_difference = 0.2  # Maximum allowable difference between body and barrier height

    # Find the middle frame index
    middle_frame_index = len(motion) // 2

    # Iterate through frames
    for frame_index, joints in enumerate(motion):
        # Get the coordinates of the hip for the current frame
        hip = joints.get("hip", None)

        # Ensure the hip is present
        if hip is not None:
            # Use provided function to calculate the height of the person's body above the ground
            height_above_ground = hip[1]  # Assuming y-coordinate represents height

            # Calculate the allowable height based on the barrier height
            allowable_height = barrier_height + max_allowable_height_difference

            # Add constraint to total error for frames around the middle frame
            if frame_index == middle_frame_index:
                total_error += max(height_above_ground - allowable_height, 0)
            else:
                total_error += max(allowable_height - height_above_ground, 0)

            # You may add more constraints based on specific requirements

    return total_error

# Example usage:
barrier_height = 1.0  # Replace with the actual height of the barrier

motion_data = [
    {"hip": (x1, y1, z1), ...},  # Frame 0
    {"hip": (x2, y2, z2), ...},  # Frame 1
    # ... (more frames)
    {"hip": (xm, ym, zm), ...},  # Middle frame (Frame m)
    # ... (more frames)
    {"hip": (xn, yn, zn), ...},  # Last frame (Frame n)
]

# Calculate the total error
total_error_jump = compute_jump_error(motion_data, barrier_height)
print("Total Jump Error:", total_error_jump)
