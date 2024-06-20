import mat73
import numpy as np
from scipy.io import savemat

load = mat73.loadmat("./exp-1.mat")
results = load['results']

nn = 32

numB = np.zeros(nn)
for ni in range(nn): # the machine crash when ni=32...
    print('to ',ni)
    nidx = np.arange(2**ni+1) * 2**(nn-ni)
    nresult = results[nidx]
    numB[ni] = np.sum((nresult[:-1] != nresult[1:]).astype(float))

savemat("./exp-1-numB.mat", {"numB": numB})