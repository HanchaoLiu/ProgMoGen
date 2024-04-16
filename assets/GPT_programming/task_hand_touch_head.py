
# PROMPT:
# Task: a person uses his left hand to touch his head. The radius of the head is 0.1 meter. What geometric constraint need to be satisfied?
'''
Contact Constraint:
The left hand should make contact with the head. This means the Euclidean distance between the left hand and any point on the head's surface should be less than or equal to the radius of the head.
'''

# PROMPT:
# Task: a person uses his left hand to touch his head. The radius of the head is 0.1 meter. Considering the geometric constraint, write the error function code.

def compute_error(motion, head_radius):
    total_error = 0

    # Constants and thresholds
    max_allowable_distance = 0.05  # Maximum allowable distance between left hand and head surface

    # Iterate through frames
    for frame_index, joints in enumerate(motion):
        # Get the left hand and head coordinates for the current frame
        left_hand = joints.get("left_hand", None)
        head = joints.get("head", None)

        if left_hand is not None and head is not None:
            # Geometric constraint: Distance between left hand and head surface
            distance_left_hand_to_head = DistToSurface(left_hand, head, head_radius)

            # Add constraint to total error
            total_error += max(distance_left_hand_to_head - max_allowable_distance, 0)

            # You may add more constraints based on specific requirements

    return total_error

# Example usage:
motion_data = [
    {"left_hand": (x1, y1, z1), "head": (hx1, hy1, hz1), ...},  # Frame 0
    {"left_hand": (x2, y2, z2), "head": (hx2, hy2, hz2), ...},  # Frame 1
    # ... (more frames)
    {"left_hand": (xn, yn, zn), "head": (hxn, hyn, hzn), ...},  # Last frame (Frame n)
]

head_radius = 0.1  # Radius of the head

total_error = compute_error(motion_data, head_radius)
print("Total Error:", total_error)


# write the code for DistToSurface
import numpy as np

def DistToSurface(point, head, head_radius):
    # Calculate the distance between the point and the surface of the head
    vector_from_head_to_point = np.array(point) - np.array(head)
    distance_to_surface = np.linalg.norm(vector_from_head_to_point) - head_radius
    return max(distance_to_surface, 0)

# Example usage:
head_radius = 0.1
head_coordinates = (0, 0, 1)  # Replace with the actual coordinates of the head in the motion data
left_hand_coordinates = (0.05, 0.1, 0.9)  # Replace with the actual coordinates of the left hand in the motion data

distance_to_surface = DistToSurface(left_hand_coordinates, head_coordinates, head_radius)
print("Distance to Head Surface:", distance_to_surface)
