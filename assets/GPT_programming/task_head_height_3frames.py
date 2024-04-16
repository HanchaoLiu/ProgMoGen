
# PROMPT:
# Task: a person is walking. His head height is constrained to be h0 at first frame, h1 at the middle frame, and h2 at the last frame. Write the error code.

def compute_error(motion, h0, h1, h2):
    total_error = 0

    # Constants and thresholds
    num_frames = len(motion)
    mid_frame_index = num_frames // 2

    # Iterate through frames
    for frame_index, joints in enumerate(motion):
        # Get the head coordinates for the current frame
        head = joints.get("head", None)

        if head is not None:
            # Geometric constraint: Head height for keyframes
            if frame_index == 0:
                target_height = h0
            elif frame_index == mid_frame_index:
                target_height = h1
            elif frame_index == num_frames - 1:
                target_height = h2
            else:
                target_height = None

            if target_height is not None:
                height_error = MSE(head[2], target_height)
                total_error += height_error

            # You may add more constraints based on specific requirements

    return total_error

def MSE(a, b):
    # Mean Squared Error between values a and b
    return (a - b) ** 2

# Example usage:
motion_data = [
    {"head": (x1, y1, z1), ...},  # Frame 0
    {"head": (x2, y2, z2), ...},  # Frame 1
    # ... (more frames)
    {"head": (xn, yn, zn), ...},  # Last frame (Frame n)
]

# Specify the constrained head heights for the first, middle, and last frames
h0 = 1.6
h1 = 1.8
h2 = 1.6

total_error = compute_error(motion_data, h0, h1, h2)
print("Total Error:", total_error)

