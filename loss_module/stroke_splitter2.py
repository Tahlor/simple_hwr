
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
#
import numpy as np
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
def seg_intersect(a1,a2, b1,b2) :
    a1,a2,b1,b2 = np.asarray(a1).astype(np.float64),\
                  np.asarray(a2).astype(np.float64),\
                  np.asarray(b1).astype(np.float64),\
                  np.asarray(b2).astype(np.float64)
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    intersection = (num / denom.astype(float))*db + b1
    # np.max(a1[1],a2[1],b1[1],b2[1]) < intersection[1]
    return intersection


# .5 in the middle
a1 = [0,0]
a2 = [1,1]
b1 = [1,0]
b2 = [0,1]
print(seg_intersect(a1,a2,b1,b2))

# 1,1 - endpoint
a1 = [0, 0]
a2 = [1, 1]
b1 = [.5, .6]
b2 = [1, 1]
print(seg_intersect(a1,a2,b1,b2))

# no intersection
a1 = [0, 0]
a2 = [1, 1]
b1 = [.5, .5]
b2 = [3, 3]
print(seg_intersect(a1,a2,b1,b2))

# no intersection??
a1 = [0, 0]
a2 = [1, 1]
b1 = [3, 2]
b2 = [4, 1]
print(seg_intersect(a1,a2,b1,b2))