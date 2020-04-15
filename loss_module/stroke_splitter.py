import matplotlib.pyplot as plt
import numpy as np

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def check_colinear(A,B,C,D):
    # Vertical or same slope
    slope1 = (A[1] - B[1]) / (A[0] - B[0])
    slope2 = (C[1] - D[1]) / (C[0] - D[0])
    vertical = C[0] == D[0] == A[0] == B[0]
    parallel = slope1==slope2 or vertical
    if vertical:
        same_intercept = A[1]==C[1]
    else:
        same_intercept = slope1
    overlapping = max(A[1],B[1]) > min(C[1],D[1])
    return parallel and same_intercept and overlapping

def check_same_point(A,B,C,D):
    return np.all(A==C) or np.all(A==D) or np.all(B==C) or np.all(B==D)

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return (ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)) or check_colinear(A,B,C,D) or check_same_point(A,B,C,D)

if __name__=='__main__':
    a1 = [0,0]
    a2 = [1,1]
    b1 = [1,0]
    b2 = [0,1]

    a1 = [0, 0]
    a2 = [1, 1]
    b1 = [.5, .6]
    b2 = [1, 1]

    print(intersect(a1,a2,b1,b2))
    # plt.scatter([a1,a2])
    # plt.scatter([b1,b2])
    # plt.show()