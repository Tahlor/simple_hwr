from hwr_utils import stroke_recovery
from matplotlib import pyplot as plt
### Go for each point, look at every possible intersection more than 10 away
###


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
def seg_intersect(a1,a2, b1,b2, threshold=0): # effectively doubled since it happens on both lines
    """ Adding a threshold may duplicate intersections

    Args:
        a1:
        a2:
        b1:
        b2:
        threshold:

    Returns:

    """
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
    #locals().update(globals())
    if any(np.isnan((intersection))):
        return None
    # elif np.amax([a1[1],a2[1]]) < intersection[1] or np.amax([b1[1],b2[1]]) < intersection[1]\
    #         or np.amin([a1[1], a2[1]]) > intersection[1] or np.amin([b1[1], b2[1]]) > intersection[1]:
    elif not np.amin([a1[1], a2[1]]) - threshold <= intersection[1] <= np.amax([a1[1], a2[1]]) + threshold or \
         not np.amin([b1[1], b2[1]]) - threshold <= intersection[1] <= np.amax([b1[1], b2[1]]) + threshold:
        return None
    else:
        return intersection

def find_loops_stroke(gt, offset=20):
    """

    Args:
        gt: Just for a single stroke
        offset:

    Returns:

    """

    gt = np.asarray(gt)[:,:2]
    coords = []
    indices = []
    for i in range(len(gt)):
        start = i + 2 + offset
        for j in range(len(gt[start:])):
            if j+start+1 >= len(gt): # need at least 2 points to define a segment
                break
            result = seg_intersect(gt[i],gt[i+1],gt[start+j],gt[start+j+1])
            if not result is None:
                #print(i,j+start, result)
                #print(gt[i],gt[i+1],gt[j],gt[j+1])
                indices.append((i,start+j))
                coords.append(result)
    if coords:
        return coords, indices
    else:
        return None, None

def find_loops_strokes(gt, offset=20, stroke_numbers=True):
    sos_args = np.append(stroke_recovery.get_sos_args(gt[:,2], stroke_numbers=stroke_numbers), len(gt))
    for i in range(len(sos_args)-1): # loop through strokes
        st = sos_args[i]
        end = sos_args[i+1]
        coords, indices = find_loops_stroke(gt[st:end], offset=offset)
        if coords:
            indices = np.asarray(indices) + st
            yield coords, indices

def test():
    gt = np.array(range(36)).reshape(9, 4).astype(np.float64)
    gt[:, 2] = [1, 0, 0, 1, 0, 1, 1, 0, 0]

    gt =    [[8, 9, 1, 3],
             [4, 5, 0, 7],
             [3, 1, 0, 11],
             [0, 5, 0, 15],
             [5, 3, 1, 19],
             [1, 8, 0, 23],
             [5, 4, 0, 31],
             [0, 1, 0, 35]]

    gt = np.asarray(gt)
    plt.plot(gt[:,0], gt[:,1])
    plt.show()
    print(find_loops_stroke(gt[0:8], offset=0))
    print(list(find_loops_strokes(gt, offset=0, stroke_numbers=False)))


def test_cases():
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

    # no intersection, parallel
    a1 = [0, 0]
    a2 = [1, 1]
    b1 = [.5, .5]
    b2 = [3, 3]
    print(seg_intersect(a1,a2,b1,b2))

    # no intersection
    a1 = [0, 0]
    a2 = [1, 1]
    b1 = [3, 2]
    b2 = [4, 1]
    print(seg_intersect(a1,a2,b1,b2))

    # no intersection
    a1 = [0, 0]
    a2 = [1, 1]
    b1 = [.1, 0]
    b2 = [.6, .3]
    print(seg_intersect(a1,a2,b1,b2))

if __name__=='__main__':
    test()
    #test_cases()
