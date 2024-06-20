# For additive perturbation case
# cosine perturbation
# multiple GPU for multiple function parameters

import numpy as np
from numba import cuda, float64
from concurrent.futures import ThreadPoolExecutor
import math
from scipy.io import savemat
import time

# Define device functions
@cuda.jit(device=True)
def gradf(x,delta1,wl1,delta2,wl2):
    return 2*x - 2*math.pi * delta1 * math.sin(2*math.pi*x / wl1) / wl1 - 2*math.pi * delta2 * math.sin(2*math.pi*x / wl2) / wl2

@cuda.jit(device=True)
def lossf(x,delta1,wl1,delta2,wl2):
    return x**2 + delta1 * math.cos(2*math.pi*x / wl1) + delta2 * math.cos(2*math.pi*x / wl2)

# Define a kernel that uses the device function
@cuda.jit
def GD(tensor, delta1_span, delta2_span, wl1, wl2, x, steps):
    #tensor : block num, thread num, steps
    ii = cuda.blockIdx.x
    ii1 = cuda.blockIdx.y
    ii2 = cuda.blockIdx.z
    jj = cuda.threadIdx.x
    lr = float64(1.5) / (tensor.shape[2] * tensor.shape[3] - 1) * (ii * tensor.shape[3] + jj)
    delta1 = delta1_span[ii1]
    delta2 = delta2_span[ii2]
    
    plus = 0.0
    plusinv = 0.0

    for step in range(steps):
        x = x - lr * gradf(x,delta1,wl1,delta2,wl2)
        plus = plus + lossf(x,delta1,wl1,delta2,wl2)
        plusinv = plusinv + (lossf(x,delta1,wl1,delta2,wl2))**(-1)
        
    tensor[ii1,ii2,ii,jj,0] = plus
    tensor[ii1,ii2,ii,jj,1] = plusinv

# define single gpu function
def onegpu(delta1_span, delta2_span, wl1, wl2, x,steps, nn, bound = 1e+16):
    num_grid = delta1_span.shape[0]
    numblocks = 2**nn

    tensor = np.zeros((num_grid,num_grid,numblocks, 2**10, 2),dtype=np.float64)
    delta1_spand = cuda.to_device(delta1_span)
    delta2_spand = cuda.to_device(delta2_span)
    tensord = cuda.to_device(tensor)

    # Measure time using time module
    start_time = time.time()

    GD[(numblocks,num_grid,num_grid), 2**10](tensord, delta1_spand, delta2_spand, wl1, wl2, x, steps)

    # Wait for the kernel to finish
    cuda.synchronize()

    # Measure time using time module
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    # Copy the result back to host
    tensord.copy_to_host(tensor)

    timeseries = tensor.reshape(num_grid,num_grid,-1,2)
    timeseries = np.nan_to_num(timeseries, nan=bound)

    result = np.zeros((num_grid,num_grid,numblocks*2**10,2))

    result[timeseries[:,:,:,0]>=bound, 0] = 1.0
    result[timeseries[:,:,:,0]>=bound, 1] = timeseries[timeseries[:,:,:,0]>=bound,1]

    result[timeseries[:,:,:,0]<bound, 0] = 0.0
    result[timeseries[:,:,:,0]<bound, 1] = timeseries[timeseries[:,:,:,0]<bound,0]

    return result

# Main function to initialize data and run the computation
if __name__ == '__main__':
    nn = 20
    nb = nn - 10
    steps = 1000
    x = float64(5.0)
    num_grid = 20
    delta1_span = np.linspace(0, .008,num_grid,dtype=np.float64)
    delta2_span = np.linspace(0, .02,num_grid,dtype=np.float64)
    wl1 = float64(0.3)
    wl2 = float64(0.5)

    # Run the processing on multiple GPUs in parallel
    outputs = onegpu(delta1_span, delta2_span, wl1, wl2, x,steps, nb)
    outputs = np.concatenate((np.zeros((num_grid,num_grid,1,2)),outputs),axis=2)
    
    numB = np.zeros((num_grid,num_grid,nn+1))
    for ii in range(num_grid):
        for jj in range(num_grid):
            for ni in range(nn+1):
                nidx = np.arange(2**ni+1) * 2**(nn-ni)
                nresult = outputs[ii,jj,nidx,0]
                numB[ii,jj,ni] = np.sum((nresult[:-1] != nresult[1:]).astype(float))
    
    dictosave = {"numB": numB}
    savemat("gpuexp-10.mat", dictosave)