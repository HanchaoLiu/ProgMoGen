
# PROMPT:
# task: a person is walking with his both feet on a balance beam. The balance beam is represented as a line L.  I want to minimize the distance between feet and balance beam

import math

# Sample function to compute distance between a point and a line
def DistToLine(point, line_point1, line_point2):
    # Implement your distance calculation logic here
    # This can be, for example, the perpendicular distance from the point to the line
    pass

def compute_total_distance(Motion, balance_beam_line):
    total_distance_left = 0
    total_distance_right = 0

    # Assuming Motion is a list of dictionaries, each dictionary representing 3D coordinates of body joints
    for frame, joints in enumerate(Motion):
        # Extract the coordinates of relevant joints (e.g., feet)
        left_foot = joints.get("left_foot", None)
        right_foot = joints.get("right_foot", None)

        if left_foot is not None and right_foot is not None:
            # Calculate distances to the balance beam line for both feet
            distance_left = DistToLine(left_foot, balance_beam_line[0], balance_beam_line[1])
            distance_right = DistToLine(right_foot, balance_beam_line[0], balance_beam_line[1])

            # Sum up the distances for each foot
            total_distance_left += distance_left
            total_distance_right += distance_right

    # Sum up the total distances for both feet
    total_distance = total_distance_left + total_distance_right

    return total_distance

# Example usage
Motion = [
    {"left_foot": (1, 2, 3), "right_foot": (4, 5, 6)},
    {"left_foot": (7, 8, 9), "right_foot": (10, 11, 12)},
    # Add more frames as needed
] 

balance_beam_line = [(0, 0, 0), (0, 1, 0)]  # Example line (adjust as needed)

total_distance = compute_total_distance(Motion, balance_beam_line)
print("Total distance between feet and balance beam:", total_distance)
