from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean
import numpy as np

# Time series 1: numpy array, shape = [m, d] where m = desired_num_of_strokes and d = dim
X = np.array(range(0,9)).reshape(3,3)
# Time series 2: numpy array, shape = [n, d] where n = desired_num_of_strokes and d = dim
Y = np.array(range(0,9)).reshape(3,3)[::-1]

# D can also be an arbitrary distance matrix: numpy array, shape [m, n]
D = SquaredEuclidean(X, Y)
print(D)
sdtw = SoftDTW(D, gamma=1.0)
# soft-DTW discrepancy, approaches DTW as gamma -> 0
value = sdtw.compute()
# gradient w.r.t. D, shape = [m, n], which is also the expected alignment matrix
E = sdtw.grad()
# gradient w.r.t. X, shape = [m, d]
G = D.jacobian_product(E)