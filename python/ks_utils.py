import numpy as np


def kernel_function(t,tk,h):
    # tk can be a sequence
    return 1/((2 * np.pi)**0.5 * h) * np.exp( - (t - tk)**2 / (2 * h**2))

def kernel_smooth(game_matrix_list,h,T_list = None):
    T, N = game_matrix_list.shape[0:2]
    smoothed = game_matrix_list + 0
    
    if T_list is None:
        T_list = np.arange(T)
    for t in T_list:
        matrix_this = smoothed[t,:,:]
        tt = (t + 1) / T
        tk = (np.arange(T) + 1) / T
        weight = kernel_function(tt,tk,h)
        for i in range(N):
            for j in range(N):
                matrix_this[i,j] = sum(weight * game_matrix_list[:,i,j])/sum(weight)
        smoothed[t,:,:] = matrix_this
    return smoothed[T_list,:,:]