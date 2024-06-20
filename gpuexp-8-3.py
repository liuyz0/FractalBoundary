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
def gradf(x,delta,wl):
    return 2*x*(1 + delta * math.cos(2*math.pi*x / wl)) - 2*math.pi*x**2 * delta * math.sin(2*math.pi*x / wl) / wl

@cuda.jit(device=True)
def lossf(x,delta,wl):
    return x**2 * (1 + delta * math.cos(2*math.pi*x / wl))

# Define a kernel that uses the device function
@cuda.jit
def GD(tensor, delta_span, wl_span, x, steps, bound):
    #tensor : block num, thread num, steps
    ii = cuda.blockIdx.x
    ii1 = cuda.blockIdx.y
    ii2 = cuda.blockIdx.z
    jj = cuda.threadIdx.x
    lr = float64(1.5) / (tensor.shape[2] * tensor.shape[3] - 1) * (ii * tensor.shape[3] + jj)
    delta = delta_span[ii1]
    wl = wl_span[ii2]
    
    plus = 0.0
    plusinv = 0.0

    for step in range(steps):
        x = x - lr * gradf(x,delta,wl)
        '''
        plus = plus + lossf(x,delta,wl)
        plusinv = plusinv + (lossf(x,delta,wl))**(-1)
        '''
        plus = lossf(x,delta,wl)
        plusinv = (lossf(x,delta,wl))**(-1)
        if plus >= bound:
            break
        
    tensor[ii1,ii2,ii,jj,0] = plus
    tensor[ii1,ii2,ii,jj,1] = plusinv

# define single gpu function
def onegpu(delta_span, wl_span, x,steps, nn, bound = 1e+6):
    num_grid = delta_span.shape[0]
    numblocks = 2**nn

    tensor = np.zeros((num_grid,num_grid,numblocks, 2**10, 2),dtype=np.float64)
    delta_spand = cuda.to_device(delta_span)
    wl_spand = cuda.to_device(wl_span)
    tensord = cuda.to_device(tensor)

    # Measure time using time module
    start_time = time.time()

    GD[(numblocks,num_grid,num_grid), 2**10](tensord, delta_spand, wl_spand, x, steps, bound)

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
    num_grid = 10
    delta_span = 10**np.linspace(-10,-2,num_grid,dtype=np.float64)
    wl_span = np.linspace(0.01,1.0,num_grid,dtype=np.float64)

    # Run the processing on multiple GPUs in parallel
    outputs = onegpu(delta_span, wl_span, x,steps, nb)
    outputs = np.concatenate((np.zeros((num_grid,num_grid,1,2)),outputs),axis=2)
    
    numB = np.zeros((num_grid,num_grid,nn+1))
    for ii in range(num_grid):
        for jj in range(num_grid):
            for ni in range(nn+1):
                nidx = np.arange(2**ni+1) * 2**(nn-ni)
                nresult = outputs[ii,jj,nidx,0]
                numB[ii,jj,ni] = np.sum((nresult[:-1] != nresult[1:]).astype(float))
    
    dictosave = {"numB": numB}
    savemat("gpuexp-8-3.mat", dictosave)
