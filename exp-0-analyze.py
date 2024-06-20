import mat73
import numpy as np
from scipy.io import savemat

load = mat73.loadmat("./exp-0.mat")
results = load['results']

nn = 32

numB = np.zeros(nn+1)
for ni in range(nn+1): # the machine crash when ni=32...
    print('to ',ni)
    if ni < nn:
        nidx = np.arange(2**ni+1) * 2**(nn-ni)
        nresult = results[nidx]
        numB[ni] = np.sum((nresult[:-1] != nresult[1:]).astype(float))
    else:
        numB[ni] = np.sum((results[:-1] != results[1:]).astype(float))

savemat("./exp-0-numB.mat", {"numB": numB})