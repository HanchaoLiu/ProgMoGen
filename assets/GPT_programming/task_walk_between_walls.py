
# PROMPT:
# walk between two walls.
# Task: a person is walking in the gap between two narrow walls. The gap can be represented as -0.2 < x < 0.2

import math

# Sample function to compute distance between a point and a plane (representing a wall)
def DistToPlane(point, wall_x_min, wall_x_max):
    # Implement your distance calculation logic here
    # This can be, for example, the perpendicular distance from the point to the wall
    x_coord = point[0]
    if x_coord < wall_x_min:
        distance = wall_x_min - x_coord
    elif x_coord > wall_x_max:
        distance = x_coord - wall_x_max
    else:
        distance = 0.0  # Inside the gap

    return distance

def compute_total_distance_to_walls(Motion, wall_x_min, wall_x_max):
    total_distance = 0

    # Assuming Motion is a list of dictionaries, each dictionary representing 3D coordinates of body joints
    for frame, joints in enumerate(Motion):
        # Extract the coordinates of relevant joints
        for joint_name, joint_coords in joints.items():
            # Calculate distances to the walls for each joint
            distance_to_wall = DistToPlane(joint_coords, wall_x_min, wall_x_max)

            # Add the distance to the total distance
            total_distance += distance_to_wall

    return total_distance

# Example usage
Motion = [
    {"head": (-0.1, 1, 3), "left_hand": (-0.2, 1, 3), "right_hand": (0.1, 1, 3)},
    {"head": (0.2, 2, 3), "left_hand": (-0.3, 2, 3), "right_hand": (0.2, 2, 3)},
    # Add more frames as needed
]

wall_x_min = -0.2
wall_x_max = 0.2

total_distance_to_walls = compute_total_distance_to_walls(Motion, wall_x_min, wall_x_max)
print("Total distance to walls:", total_distance_to_walls)

