import numpy as np

def point_to_line_distance(A, B, P):
    AB = B - A
    AP = P - A
    BP = P - B
    # Scalar projection
    t = np.dot(AP, AB) / np.dot(AB, AB)
    if t < 0:
        # Closest point is A
        d = np.linalg.norm(AP)
    elif t > 1:
        # Closest point is B
        d = np.linalg.norm(BP)
    else:
        # Closest point is on the segment
        d = np.linalg.norm(AP - t * AB)

    return d