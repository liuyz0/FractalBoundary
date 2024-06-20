# high dim multiplicative

import numpy as np
from scipy.io import savemat

def lossf(x,delta,wl):
    return np.sum(x**2) * (1 + delta * np.sum(np.cos(2*np.pi*x / wl)))

def gradf(x,delta,wl):
    return 2*x*(1 + delta * np.sum(np.cos(2*np.pi*x / wl))) - 2*np.pi*np.sum(x**2) * delta * np.sin(2*np.pi*x / wl) / wl

def GD(d,x0, lr, steps, delta, wl):
    x_traj = np.zeros((steps+1,d))
    x_traj[0] = x0
    loss_traj = - np.ones(steps+1)
    loss_traj[0] = lossf(x0,delta,wl)
    for i in range(steps):
        x_traj[i+1] = x0 - lr * gradf(x0,delta,wl)
        x0 = x_traj[i+1]
        loss_traj[i+1] = lossf(x0,delta,wl)
        if loss_traj[i+1] > 1e+8 * d:
            loss_traj[i+1] = -1
            break
    return loss_traj, x_traj

nn = 16
d_span = np.arange(1,101)
s_span = np.linspace(0,1.5,2**nn+1)
results = - np.ones((len(d_span),2**nn+1)) # 0 must be -1

for ii in range(len(d_span)):
    d = d_span[ii]
    print('to d=', d)
    for i in range(2**nn):
        # 0 already known
        s = s_span[i+1]
        loss_traj, _ = GD(d,np.ones(d)/d, s, 1000, .2, .1) # smaller wave length!!
        if loss_traj[-1] == -1:
            results[ii, i+1] = 1

savemat("./exp-3.mat", {"results": results})