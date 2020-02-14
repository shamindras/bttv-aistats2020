import sys
import numpy as np
import scipy as sc
import scipy.linalg as spl
import grad_utils as model
import ks_utils as ks
import simulation_utils as si
import opt_utils as op

def loocv(data, lambdas_smooth, opt_fn,
          num_loocv = 200, get_estimate = True, 
          verbose = 'cv', out = 'terminal', **kwargs):
    '''
    conduct local
    ----------
    Input:
    data: TxNxN array
    lambdas_smooth: a vector of query lambdas
    opt_fn: a python function in a particular form of 
        opt_fn(data, lambda_smooth, beta_init=None, **kwargs)
        kwargs might contain hyperparameters 
        (e.g., step size, max iteration, etc.) for
        the optimization function
    num_loocv: the number of random samples left-one-out cv sample
    get_estimate: whether or not we calculate estimates beta's for 
        every lambdas_smooth. If True, we use those estimates as 
        initial values for optimizations with cv data
    verbose: controlling the verbose level. If 'cv', the function 
        prints only cv related message. If 'all', the function prints
        all messages including ones from optimization process.
        The default is 'cv'.
    out: controlling the direction of output. If 'terminal', the function
        prints into the terminal. If 'notebook', the function prints into 
        the ipython notebook. If 'file', the function prints into a log 
        file 'cv_log.txt' at the same directory. You can give a custom 
        output stream to this argument. The default is 'terminal'
    **kwargs: keyword arguments for opt_fn
    ----------
    Output:
    lambda_cv: lambda_smooth chosen after cross-validation
    nll_cv: average cross-validated negative loglikelihood 
    beta_cv: beta chosen after cross-validation. None if get_estimate is False
    '''    
    lambdas_smooth = lambdas_smooth.flatten()
    lambdas_smooth = -np.sort(-lambdas_smooth)
    betas = [None] * lambdas_smooth.shape[0]
    
    last_beta = np.zeros(data.shape[:2])
    for i, lambda_smooth in enumerate(lambdas_smooth):
        _, beta = opt_fn(data, lambda_smooth, beta_init = last_beta, **kwargs)
        betas[i] = beta.reshape(data.shape[:2])
        last_beta = betas[i]
        
    indices = np.array(np.where(np.full(data.shape, True))).T
    cum_match = np.cumsum(data.flatten())
    
    if out == 'terminal':
        out = sys.__stdout__
    elif out == 'notebook':
        out = sys.stdout
    elif out == 'file':
        out = open('cv_log.txt', 'w')
    
    loglikes_loocv = np.zeros(lambdas_smooth.shape)
    for i in range(num_loocv):
        data_loocv = data.copy()
        rand_match = np.random.randint(np.sum(data))
        rand_index = indices[np.min(np.where(cum_match >= rand_match)[0])]
        data_loocv[tuple(rand_index)] -= 1

        for j, lambda_smooth in enumerate(lambdas_smooth):
            _, beta_loocv = opt_fn(data_loocv, lambda_smooth, beta_init=betas[j],
                                   verbose=(verbose in ['all']), out=out, **kwargs)
            beta_loocv = beta_loocv.reshape(data.shape[:2])
            loglikes_loocv[j] += beta_loocv[rand_index[0],rand_index[1]] \
                   - np.log(np.exp(beta_loocv[rand_index[0],rand_index[1]])
                            + np.exp(beta_loocv[rand_index[0],rand_index[2]]))
        
        if verbose in ['cv', 'all']:
            out.write("%d-th cv done\n"%(i+1))
            out.flush()
        
    return (lambdas_smooth[np.argmax(loglikes_loocv)], -loglikes_loocv[::-1]/num_loocv, 
            betas[np.argmax(loglikes_loocv)])



def loocv_ks(data, h_list, opt_fn,
          num_loocv = 200, get_estimate = True, return_prob = True,
          verbose = 'cv', out = 'terminal', **kwargs):
    '''
    conduct local
    ----------
    Input:
    data: TxNxN array
    h: a vector of kernel parameters
    opt_fn: a python function in a particular form of 
        opt_fn(data, lambda_smooth, beta_init=None, **kwargs)
        kwargs might contain hyperparameters 
        (e.g., step size, max iteration, etc.) for
        the optimization function
    num_loocv: the number of random samples left-one-out cv sample
    get_estimate: whether or not we calculate estimates beta's for 
        every lambdas_smooth. If True, we use those estimates as 
        initial values for optimizations with cv data
    verbose: controlling the verbose level. If 'cv', the function 
        prints only cv related message. If 'all', the function prints
        all messages including ones from optimization process.
        The default is 'cv'.
    out: controlling the direction of output. If 'terminal', the function
        prints into the terminal. If 'notebook', the function prints into 
        the ipython notebook. If 'file', the function prints into a log 
        file 'cv_log.txt' at the same directory. You can give a custom 
        output stream to this argument. The default is 'terminal'
    **kwargs: keyword arguments for opt_fn
    ----------
    Output:
    lambda_cv: lambda_smooth chosen after cross-validation
    nll_cv: average cross-validated negative loglikelihood 
    beta_cv: beta chosen after cross-validation. None if get_estimate is False
    '''    
    h_list = h_list.flatten()
    h_list = -np.sort(-h_list)
    betas = [None] * h_list.shape[0]
    
    last_beta = np.zeros(data.shape[:2])
    for i, h in enumerate(h_list):
        
        ks_data = ks.kernel_smooth(data,h)
        _, beta = opt_fn(ks_data, beta_init = last_beta, **kwargs)
        betas[i] = beta.reshape(data.shape[:2])
        last_beta = betas[i]
        
    indices = np.array(np.where(np.full(data.shape, True))).T
    cum_match = np.cumsum(data.flatten())
    
    if out == 'terminal':
        out = sys.__stdout__
    elif out == 'notebook':
        out = sys.stdout
    elif out == 'file':
        out = open('cv_log.txt', 'w')
    
    loglikes_loocv = np.zeros(h_list.shape)
    prob_loocv = np.zeros(h_list.shape)
    for i in range(num_loocv):
        data_loocv = data.copy()
        rand_match = np.random.randint(np.sum(data))
        rand_index = indices[np.min(np.where(cum_match >= rand_match)[0])]
        data_loocv[tuple(rand_index)] -= 1

        for j, h in enumerate(h_list):
            ks_data_loocv = ks.kernel_smooth(data_loocv,h)
            _, beta_loocv = opt_fn(ks_data_loocv, beta_init=betas[j],
                                   verbose=(verbose in ['all']), out=out, **kwargs)
            beta_loocv = beta_loocv.reshape(data.shape[:2])
            loglikes_loocv[j] += beta_loocv[rand_index[0],rand_index[1]] \
                   - np.log(np.exp(beta_loocv[rand_index[0],rand_index[1]])
                            + np.exp(beta_loocv[rand_index[0],rand_index[2]]))
            prob_loocv[j] += 1 - np.exp(beta_loocv[rand_index[0],rand_index[1]]) \
                / (np.exp(beta_loocv[rand_index[0],rand_index[1]])
                    + np.exp(beta_loocv[rand_index[0],rand_index[2]]))

        if verbose in ['cv', 'all']:
            out.write("%d-th cv done\n"%(i+1))
            out.flush()
    if return_prob:
        return (h_list[np.argmax(loglikes_loocv)], -loglikes_loocv[::-1]/num_loocv, 
                betas[np.argmax(loglikes_loocv)], prob_loocv[::-1]/num_loocv)
    else:
        return (h_list[np.argmax(loglikes_loocv)], -loglikes_loocv[::-1]/num_loocv, 
                betas[np.argmax(loglikes_loocv)])
    
    
def loo_DBT(data, h, opt_fn,
          num_loo = 200, get_estimate = True, return_prob = True,
          verbose = 'cv', out = 'terminal', **kwargs):
    '''
    conduct local
    ----------
    Input:
    data: TxNxN array
    h: a vector of kernel parameters
    opt_fn: a python function in a particular form of 
        opt_fn(data, lambda_smooth, beta_init=None, **kwargs)
        kwargs might contain hyperparameters 
        (e.g., step size, max iteration, etc.) for
        the optimization function
    num_loo: the number of random samples left-one-out sample
    get_estimate: whether or not we calculate estimates beta's for 
        every lambdas_smooth. If True, we use those estimates as 
        initial values for optimizations with cv data
    verbose: controlling the verbose level. If 'cv', the function 
        prints only cv related message. If 'all', the function prints
        all messages including ones from optimization process.
        The default is 'cv'.
    out: controlling the direction of output. If 'terminal', the function
        prints into the terminal. If 'notebook', the function prints into 
        the ipython notebook. If 'file', the function prints into a log 
        file 'cv_log.txt' at the same directory. You can give a custom 
        output stream to this argument. The default is 'terminal'
    **kwargs: keyword arguments for opt_fn
    ----------
    Output:
    
    '''    
    
    last_beta = np.zeros(data.shape[:2])
        
    ks_data = ks.kernel_smooth(data,h)
    _, beta = opt_fn(ks_data, beta_init = last_beta, **kwargs)
    beta = beta.reshape(data.shape[:2])
    
    indices = np.array(np.where(np.full(data.shape, True))).T
    cum_match = np.cumsum(data.flatten())
    
    if out == 'terminal':
        out = sys.__stdout__
    elif out == 'notebook':
        out = sys.stdout
    elif out == 'file':
        out = open('cv_log.txt', 'w')
    
    loglikes_loo = 0
    prob_loo = 0
    for i in range(num_loo):
        data_loo = data.copy()
        rand_match = np.random.randint(np.sum(data))
        rand_index = indices[np.min(np.where(cum_match >= rand_match)[0])]
        data_loo[tuple(rand_index)] -= 1

        ks_data_loo = ks.kernel_smooth(data_loo,h)
        _, beta_loo = opt_fn(ks_data_loo, beta_init=beta,
                               verbose=(verbose in ['all']), out=out, **kwargs)
        beta_loo = beta_loo.reshape(data.shape[:2])
        loglikes_loo += beta_loo[rand_index[0],rand_index[1]] \
               - np.log(np.exp(beta_loo[rand_index[0],rand_index[1]])
                        + np.exp(beta_loo[rand_index[0],rand_index[2]]))
        prob_loo += 1 - np.exp(beta_loo[rand_index[0],rand_index[1]]) \
            / (np.exp(beta_loo[rand_index[0],rand_index[1]])
                + np.exp(beta_loo[rand_index[0],rand_index[2]]))

        if verbose in ['cv', 'all']:
            out.write("%d-th cv done\n"%(i+1))
            out.flush()
    if return_prob:
        return (-loglikes_loo/num_loo, 
                beta, prob_loo/num_loo)
    else:
        return (-loglikes_loo/num_loo, 
                beta)




def loo_vBT(data,num_loo = 200):
    T,N = data.shape[:2]
    _, beta = op.gd_bt(data = data)
    
    indices = np.array(np.where(np.full(data.shape, True))).T
    cum_match = np.cumsum(data.flatten())
    
    loglikes_loo = 0
    prob_loo = 0
    for i in range(num_loo):
        data_loo = data.copy()
        beta_loo = beta.copy()
        rand_match = np.random.randint(np.sum(data))
        rand_index = indices[np.min(np.where(cum_match >= rand_match)[0])]
        data_loo[tuple(rand_index)] -= 1
        data_loo = data_loo[rand_index[0]].reshape((1,N,N))
        
        beta_loo_i = beta[rand_index[0],:]
        _, beta_loo_i = op.gd_bt(data = data_loo,beta_init = beta_loo_i)
        beta_loo[rand_index[0]] = beta_loo_i
        beta_loo_i = beta_loo_i.reshape((N,))
        loglikes_loo += beta_loo_i[rand_index[1]] \
                   - np.log(np.exp(beta_loo_i[rand_index[1]])
                            + np.exp(beta_loo_i[rand_index[2]]))
        prob_loo += 1 - np.exp(beta_loo_i[rand_index[1]]) \
            / (np.exp(beta_loo_i[rand_index[1]])
                + np.exp(beta_loo_i[rand_index[2]]))

    return (-loglikes_loo/num_loo, prob_loo/num_loo)


def loo_winrate(data,num_loo = 200):
    indices = np.array(np.where(np.full(data.shape, True))).T
    cum_match = np.cumsum(data.flatten())
    
    loglikes_loo = 0
    prob_loo = 0
    for i in range(num_loo):
        data_loo = data.copy()
        rand_match = np.random.randint(np.sum(data))
        rand_index = indices[np.min(np.where(cum_match >= rand_match)[0])]
        data_loo[tuple(rand_index)] -= 1
        
        winrate_loo = si.get_winrate(data = data_loo)
        prob_loo += 1 - winrate_loo[rand_index[0],rand_index[1]]

    return (-loglikes_loo/num_loo, prob_loo/num_loo)