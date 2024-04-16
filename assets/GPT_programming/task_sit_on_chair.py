
# PROMPT:
# Task: a person is sitting on a chair, considering geometric constraints, write the error function code.

import numpy as np

def compute_error(motion, chair_height):
    total_error = 0

    # Constants and thresholds
    max_allowable_distance = 0.1  # Maximum allowable distance between buttocks and chair surface
    max_allowable_height_difference = 0.1  # Maximum allowable height difference between buttocks and chair surface

    # Iterate through frames
    for frame_index, joints in enumerate(motion):
        # Get the coordinates of the buttocks and the chair for the current frame
        buttocks = joints.get("buttocks", None)
        chair_surface = joints.get("chair_surface", None)  # Replace with the actual chair surface coordinates

        if buttocks is not None and chair_surface is not None:
            # Geometric constraint: Distance between buttocks and chair surface
            distance_buttocks_to_chair = DistToSurface(buttocks, chair_surface)
            total_error += max(distance_buttocks_to_chair - max_allowable_distance, 0)

            # Geometric constraint: Height difference between buttocks and chair surface
            height_buttocks_difference = np.abs(buttocks[2] - chair_surface[2])
            total_error += max(height_buttocks_difference - max_allowable_height_difference, 0)

            # You may add more constraints based on specific requirements

    return total_error

# Example usage:
motion_data = [
    {"buttocks": (x1, y1, z1), "chair_surface": (cx1, cy1, cz1), ...},  # Frame 0
    {"buttocks": (x2, y2, z2), "chair_surface": (cx2, cy2, cz2), ...},  # Frame 1
    # ... (more frames)
    {"buttocks": (xn, yn, zn), "chair_surface": (cxn, cyn, czn), ...},  # Last frame (Frame n)
]

chair_height = 0.5  # Height of the chair

total_error = compute_error(motion_data, chair_height)
print("Total Error:", total_error)

