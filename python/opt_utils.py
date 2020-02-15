import sys
import numpy as np
import scipy as sc
import pandas as pd
import scipy.linalg as spl
import grad_utils as model



def gd_bt(data,
              max_iter=1000, ths=1e-12,
              step_init=0.5, max_back=200, a=0.2, b=0.5,
              beta_init=None, verbose=False, out=sys.stdout):
    # initialize optimization
    T, N = data.shape[0:2]
    if beta_init is None:
        beta = np.zeros(data.shape[:2])
    else:
        beta = beta_init
    nll = model.neg_log_like(beta, data)

    # initialize record
    objective_wback = [nll]
    if verbose:
        out.write("initial objective value: %f\n"%objective_wback[-1])
        out.flush()

    # iteration
    for i in range(max_iter):
        # compute gradient
        gradient = model.grad_nl(beta, data).reshape([T,N])
        
        # backtracking line search
        s = step_init
        for j in range(max_back):
            beta_new = beta - s*gradient
            beta_diff = beta_new - beta
            
            nll_new = model.neg_log_like(beta_new, data)
            nll_back = (nll + np.sum(gradient * beta_diff) 
                        + np.sum(np.square(beta_diff)) / (2*s))
            
            if nll_new <= nll_back:
                break
            s *= b
        
        # proximal gradient update
        beta = beta_new
        nll = nll_new
        
        # record objective value
        objective_wback.append(model.neg_log_like(beta, data))
        
        if verbose:
            out.write("%d-th GD, objective value: %f\n"%(i+1, objective_wback[-1]))
            out.flush()
        if abs(objective_wback[-2] - objective_wback[-1]) < ths:
            if verbose:
                out.write("Converged!\n")
                out.flush()
            break
        elif i >= max_iter-1:
            if verbose:
                out.write("Not converged.\n")
                out.flush()

    return objective_wback, beta




########################## squared l2 penalty ############################


def objective_l2_sq(beta, game_matrix_list, l_penalty):
    '''
    compute the objective of the model (neg_log_like + l2_square)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    ----------
    Output:
    objective: negative log likelihood + squared l2 penalty
    '''
    # reshape beta into TxN array
    T, N = game_matrix_list.shape[0:2]
    beta = np.reshape(beta, [T,N])
    
    # compute l2 penalty
    l2_penalty = np.sum(np.square(beta[:-1]-beta[1:]))
    
    return model.neg_log_like(beta, game_matrix_list) + l_penalty * l2_penalty


def grad_l2_sq(beta, game_matrix_list, l):
    '''
    compute the gradient of the model (neg_log_like + l2_square)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    ----------
    Output:
    objective: negative log likelihood + squared l2 penalty
    '''
    # reshape beta into TxN array
    T, N = game_matrix_list.shape[0:2]
    beta = np.reshape(beta, [T,N])
    
    # compute l2 penalty
    l2_grad = model.grad_nl(beta, game_matrix_list)
    l2_grad[N:] += l * 2 * ((beta[1:]-beta[:-1])).reshape(((T - 1) * N, 1))
    l2_grad[:-N] += l * 2 *((beta[:-1]-beta[1:])).reshape(((T - 1) * N, 1))
    
    return  l2_grad


def hess_l2_sq(beta, game_matrix_list, l):
    '''
    compute the Hessian of the model (neg_log_like + l2_square)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    ----------
    Output:
    objective: negative log likelihood + squared l2 penalty
    '''
    # reshape beta into TxN array
    T, N = game_matrix_list.shape[0:2]
    beta = np.reshape(beta, [T,N])
    
    # compute l2 penalty
    l2_hess = model.hess_nl(beta, game_matrix_list)
    off_diag = np.array([2] + [0] * (N - 1) + [-1] + [0] * (N * (T - 1) - 1))
    l2_hess += l * 2 * sc.linalg.toeplitz(off_diag,off_diag)
    l2_hess[0:N,0:N] -= l * 2 * np.diag(np.ones(N))
    l2_hess[-N:,-N:] -= l * 2 * np.diag(np.ones(N))
    return  l2_hess

def prox_l2_sq(beta, s, l):
    '''
    proximal operator for l2-square-penalty
    '''
    n = np.array(beta).shape[0]
    
    # define banded matrix
    banded = np.block([
        [np.zeros([1,1]), (-1)*2*s*l*np.ones([1,n-1])],
        [(1+2*s*l)*np.ones([1,1]), (1+2*2*s*l)*np.ones([1,n-2]), (1+2*s*l)*np.ones([1,1])],
        [(-1)*2*s*l*np.ones([1,n-1]), np.zeros([1,1])]
    ])

    # solve banded @ beta* = beta
    return spl.solve_banded((1,1), banded, beta, True, True, False)

def newton_l2_sq(data, l_penalty=1,
                 max_iter=1000, ths=1e-12,
                 step_init=1, max_back=200, a=0.01, b=0.3,
                 beta_init=None, verbose=False, out=sys.stdout):
    # initialize optimization
    T, N = data.shape[0:2]
    if beta_init is None:
        beta = np.zeros(data.shape[:2]).reshape((N * T,1))
    else:
        beta = beta_init.reshape((N*T, 1))
    
    # initialize record
    objective_nt = [objective_l2_sq(beta, data, l_penalty)]
    if verbose:
        out.write("initial objective value: %f\n"%objective_nt[-1])
        out.flush()

    # iteration
    for i in range(max_iter):
        # compute gradient
        gradient = grad_l2_sq(beta, data, l_penalty)[1:]
        hessian = hess_l2_sq(beta, data, l_penalty)[1:,1:]
        # backtracking
        obj_old = np.inf
        s = step_init
        beta_new = beta - 0 # make a copy
        
        for j in range(max_back):
            v = -sc.linalg.solve(hessian, gradient)
            beta_new[1:] = beta_new[1:] + s * v
            obj_new = objective_l2_sq(beta_new,data,l_penalty)
        
            if obj_new <= obj_old + b * s * gradient.T @ v:
                break
            s *= a
            
        beta = beta_new
        
        # objective value
        objective_nt.append(obj_new)
        obj_old = obj_new

        if verbose:
            out.write("%d-th Newton, objective value: %f\n"%(i+1, objective_nt[-1]))
            out.flush()
        if objective_nt[-2] - objective_nt[-1] < ths:
            if verbose:
                out.write("Converged!\n")
                out.flush()
            break
        elif i >= max_iter-1:
            if verbose:
                out.write("Not converged.\n")
                out.flush()

    beta = beta.reshape((T,N))
    beta = beta - sum(beta[0,0:N]) / N   

    return objective_nt, beta

def pgd_l2_sq(data, l_penalty=1,
              max_iter=1000, ths=1e-12,
              step_init=0.5, max_back=200, a=0.2, b=0.5,
              beta_init=None, verbose=False, out=sys.stdout):
    # initialize optimization
    T, N = data.shape[0:2]
    if beta_init is None:
        beta = np.zeros(data.shape[:2])
    else:
        beta = beta_init
    nll = model.neg_log_like(beta, data)

    # initialize record
    objective_wback = [objective_l2_sq(beta, data, l_penalty)]
    if verbose:
        out.write("initial objective value: %f\n"%objective_wback[-1])
        out.flush()

    # iteration
    for i in range(max_iter):
        # compute gradient
        gradient = model.grad_nl(beta, data).reshape([T,N])
        
        # backtracking line search
        s = step_init
        for j in range(max_back):
            beta_new = prox_l2_sq(beta - s*gradient, s, l_penalty)
            beta_diff = beta_new - beta
            
            nll_new = model.neg_log_like(beta_new, data)
            nll_back = (nll + np.sum(gradient * beta_diff) 
                        + np.sum(np.square(beta_diff)) / (2*s))
            
            if nll_new <= nll_back:
                break
            s *= b
        
        # proximal gradient update
        beta = beta_new
        nll = nll_new
        
        # record objective value
        objective_wback.append(objective_l2_sq(beta, data, l_penalty))
        
        if verbose:
            out.write("%d-th PGD, objective value: %f\n"%(i+1, objective_wback[-1]))
            out.flush()
        if abs(objective_wback[-2] - objective_wback[-1]) < ths:
            if verbose:
                out.write("Converged!\n")
                out.flush()
            break
        elif i >= max_iter-1:
            if verbose:
                out.write("Not converged.\n")
                out.flush()

    return objective_wback, beta

########################## l2 penalty ############################
def objective_l2(beta, game_matrix_list, l_penalty):
    '''
    compute the objective of the model (neg_log_like + l2)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    l: coefficient of penalty term
    ----------
    Output:
    objective: negative log likelihood + l2 penalty
    '''
    # reshape beta into TxN array
    T, N = game_matrix_list.shape[0:2]
    beta = np.reshape(beta, [T,N])
    
    # compute l2 penalty
    diff = beta[:-1]-beta[1:]
    l2_penalty = sum([np.linalg.norm(diff[i]) for i in range(T - 1)])
    
    return model.neg_log_like(beta, game_matrix_list) + l_penalty * l2_penalty


def grad_l2(beta, game_matrix_list, l):
    '''
    compute the gradient of the model (neg_log_like + l2)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    l: coefficient of penalty term
    ----------
    Output:
    objective: negative log likelihood + l2 penalty
    '''
    # reshape beta into TxN array
    T, N = game_matrix_list.shape[0:2]
    beta = np.reshape(beta, [T,N])
    
    # compute l2 penalty
    l2_grad = model.grad_nl(beta, game_matrix_list)
    diff = beta[1:] - beta[:-1]
    w = np.array([np.linalg.norm(diff[i]) for i in range(T - 1)])
    w[w != 0] = 1 / w[w != 0]
    l2_grad[N:] += l * np.array([diff[i] * w[i] for i in range(T - 1)]).reshape(((T - 1) * N, 1))
    
    diff = beta[:-1] - beta[1:]
    w = np.array([np.linalg.norm(diff[i]) for i in range(T - 1)])
    w[w != 0] = 1 / w[w != 0]
    l2_grad[:-N] += l * np.array([diff[i] * w[i] for i in range(T - 1)]).reshape(((T - 1) * N, 1))
    
    return  l2_grad



def gd_l2(data, l_penalty=1,
          max_iter=1000, ths=1e-12,
          step_init=1, max_back=200, a=0.01, b=0.3,
          beta_init=None, verbose=False, out=sys.stdout):
    # initialize optimization
    T, N = data.shape[0:2]
    if beta_init is None:
        beta = np.zeros(data.shape[:2])
    else:
        beta = beta_init

    # initialize record
    objective_gd = [objective_l2(beta, data, l_penalty)]
    if verbose:
        out.write("initial objective value: %f\n"%objective_gd[-1])
        out.flush()

        # iteration
    for i in range(max_iter):
        # compute gradient
        gradient = grad_l2(beta, data, l_penalty).reshape([T,N])
        
        # backtracking
        s = step_init    
        for j in range(max_back):
        # gradient update
            beta_new = beta - s * gradient
            obj_new = objective_l2(beta_new, data, l_penalty)
            if obj_new < objective_gd[-1] - b * s * np.sum(np.square(gradient)):
                break
            s *= a
        
        # objective value
        beta = beta_new
        objective_gd.append(obj_new)
        
        out.write("%d-th GD, objective value: %f\n"%(i+1, objective_gd[-1]))
        out.flush()
        if abs(objective_gd[-2] - objective_gd[-1]) < ths:
            if verbose:
                out.write("Converged!\n")
                out.flush()
            break
        elif i >= max_iter - 1:
            if verbose:
                out.write("Not converged.\n")
                out.flush()

    beta = beta.reshape((T,N))
    beta = beta - sum(beta[0,0:N]) / N
    
    return objective_gd, beta


############## functions for ADMM ################

def obj_amlag(beta,data,A,muk,eta,thetak):
    return model.neg_log_like(beta, data) + (A @ beta).T @ muk + eta / 2 * np.linalg.norm(A @ beta - thetak) ** 2
    
def admm_sub_beta(data,T,N,A,lam,eta,beta,muk,thetak,paras,out=sys.stdout):
    step_init, ths, max_iter, max_back, a, b = paras
    obj_old = np.inf

    for i in range(max_iter):    
        # compute gradient
        gradient = model.grad_nl(beta, data) + A.T @ muk + eta * A.T @ (A @ beta - thetak)
        hessian = model.hess_nl(beta, data) + eta * A.T @ A
        gradient = gradient[1:]
        hessian = hessian[1:,1:]
        # proximal gradient update
        s = step_init
        beta_new = beta - 0 # make a copy
        
        for j in range(max_back):
            v = -sc.linalg.solve(hessian, gradient)
            beta_new[1:] = beta[1:] + s * v
            obj_new = obj_amlag(beta_new,data,A,muk,eta,thetak)
        
            if obj_new <= obj_old + b * s * gradient.T @ v:
                break
            s *= a
            
        beta = beta_new
        if abs(obj_old - obj_new) < ths:
            break
        elif i >= max_iter - 1:
            out.write("Not converged.\n")
            out.flush()
        obj_old = obj_new   
    
    return beta


def prox_l2(t,x):
    if np.linalg.norm(x,2) <= t:
        return 0
    else:
        return (1 - t / np.linalg.norm(x,2)) * x

def prox_l1(t,x):
    return np.sign(x) * np.max(np.array([x * 0, np.abs(x) - t]),axis = 0)


def admm_l2(data, l_penalty=1,
            max_iter=1000, ths=1e-12, eta=None, 
            step_init=1, max_back=200, a=0.01, b=0.3,
            beta_init=None, verbose=False, out=sys.stdout, return_b_obj=False):
    # initialize optimization
    T, N = data.shape[0:2]
    if beta_init is None:
        beta = np.zeros(data.shape[:2]).reshape((N * T,1))
    else:
        beta = beta_init.reshape((N*T, 1))

    # optimization parameters
    paras = [step_init, ths, max_iter, max_back, a, b]

    A = np.zeros(((T - 1) * N, T * N))
    for i in range(N):
        for t in range(T-1):
            A[t * N + i, (t + 1) * N + i] = 1
            A[t * N + i, (t) * N + i] = -1

    if eta is None:
        eta = 20 * l_penalty

    thetak = np.zeros(((T - 1) * N,1))
    muk = A @ beta - thetak

    # initialize record
    objective_admm_b_l2 = [objective_l2(beta, data, l_penalty)]
    objective_admm = [model.neg_log_like(beta, data) + l_penalty * np.linalg.norm(thetak)]
    if verbose:
        out.write("initial objective value: %f\n"%objective_admm[-1])
        out.flush()

    # iteration
    for i in range(max_iter):
        # compute gradient
        beta = admm_sub_beta(data,T,N,A,l_penalty,eta,beta,muk,thetak,paras,out)
        thetak = prox_l2(l_penalty / eta,A @ beta + muk / eta)
        muk = muk + eta * (A @ beta - thetak)
        # objective value
        objective_admm_b_l2.append(objective_l2(beta, data, l_penalty))
        objective_admm.append(model.neg_log_like(beta, data) + l_penalty * np.linalg.norm(thetak))
        
        # if verbose
        #     print("%d-th ADMM, objective value: %f"%(i+1, objective_admm_b_l2[-1]))
        # if abs(objective_admm_b_l2[-2] - objective_admm_b_l2[-1]) < ths:
        #     print("Converged!")
        #     break
        if verbose:
            out.write("%d-th ADMM, objective value: %f\n"%(i+1, objective_admm[-1]))
            out.flush()
        if objective_admm[-2] - objective_admm[-1] < ths:
            if verbose:
                out.write("Converged!\n")
                out.flush()
            break
        elif i >= max_iter - 1:
            if verbose:
                out.write("Not converged.")
                out.flush()
            
    beta = beta - sum(beta[0:N]) / N
    beta = beta.reshape((T,N))

    if return_b_obj:
        return (objective_admm, objective_admm_b_l2), beta
    return objective_admm, beta

########################## l1 penalty ############################


def objective_l1(beta, game_matrix_list, l_penalty):
    '''
    compute the objective of the model (neg_log_like + l1)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    l: coefficient of penalty term
    ----------
    Output:
    objective: negative log likelihood + l1 penalty
    '''
    # reshape beta into TxN array
    T, N = game_matrix_list.shape[0:2]
    beta = np.reshape(beta, [T,N])
    
    # compute l2 penalty
    diff = beta[:-1]-beta[1:]
    l1_penalty = sum([np.linalg.norm(diff[i],1) for i in range(T - 1)])
    
    return model.neg_log_like(beta, game_matrix_list) + l_penalty * l1_penalty




def grad_l1(beta, game_matrix_list, l):
    '''
    compute the gradient of the model (neg_log_like + l1)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    l: coefficient of penalty term
    ----------
    Output:
    objective: negative log likelihood + l1 penalty
    '''
    # reshape beta into TxN array
    T, N = game_matrix_list.shape[0:2]
    beta = np.reshape(beta, [T,N])
    
    # compute l1 penalty
    l1_grad = model.grad_nl(beta, game_matrix_list)
    diff = beta[1:] - beta[:-1]
    l1_grad[N:] += l * np.array([np.sign(diff[i]) for i in range(T - 1)]).reshape(((T - 1) * N, 1))
    l1_grad[:-N] += l * np.array([-np.sign(diff[i]) for i in range(T - 1)]).reshape(((T - 1) * N, 1))
    
    return  l1_grad

def gd_l1(data, l_penalty=1,
          max_iter=50000, ths=1e-12, step_size=0.05, 
          beta_init=None, verbose=False, out=sys.stdout):
    # initialize optimization
    T, N = data.shape[0:2]
    if beta_init is None:
        beta = np.zeros(data.shape[:2])
    else:
        beta = beta_init

    # initialize record
    objective_gd = [objective_l1(beta, data, l_penalty)]
    if verbose:
        out.write("initial objective value: %f\n"%objective_gd[-1])
        out.flush()

    # iteration
    for i in range(max_iter):
        # compute gradient
        gradient = grad_l1(beta, data, l_penalty).reshape([T,N])
        
        # proximal gradient update
        beta = beta - step_size / round((i + 11)/10) * gradient
        
        # objective value
        objective_gd.append(objective_l1(beta, data, l_penalty))
        
        if verbose: 
            if (i + 1) % 100 == 0:
                out.write("%d-th GD, objective value: %f\n"%(i+1, objective_gd[-1]))
                out.flush()

        if abs(objective_gd[-2] - objective_gd[-1]) < ths:
            if verbose:
                out.write("Converged!\n")
                out.flush()
            break
        elif i >= max_iter - 1:
            if verbose:
                out.write("Not converged.\n")
                out.flush()

    beta = beta.reshape((T,N))
    beta = beta - sum(beta[0,0:N]) / N
            
    return objective_gd, beta


def admm_l1(data, l_penalty=1,
            max_iter=1000, ths=1e-12, eta=None,
            step_init=1, max_back=200, a=0.01, b=0.3,
            beta_init=None, verbose=False, out=sys.stdout, return_b_obj = False):
    # initialize optimization
    T, N = data.shape[0:2]
    if beta_init is None:
        beta = np.zeros(data.shape[:2]).reshape((N * T,1))
    else:
        beta = beta_init.reshape((N*T, 1))

    # optimization parameters
    paras = [step_init, ths, max_iter, max_back, a, b]

    A = np.zeros(((T - 1) * N, T * N))
    for i in range(N):
        for t in range(T-1):
            A[t * N + i, (t + 1) * N + i] = 1
            A[t * N + i, (t) * N + i] = -1

    if eta is None:
        eta = 20 * l_penalty

    thetak = np.zeros(((T - 1) * N,1))
    muk = A @ beta - thetak

    # initialize record
    objective_admm_b_l1 = [objective_l1(beta, data, l_penalty)]
    objective_admm = [model.neg_log_like(beta, data) + l_penalty 
                      * np.linalg.norm(thetak,1)]
    if verbose:
        out.write("initial objective value: %f\n"%objective_admm[-1])
        out.flush()

    # iteration
    for i in range(max_iter):
        # compute gradient
        beta = admm_sub_beta(data,T,N,A,l_penalty,eta,beta,muk,thetak,paras,out)
        thetak = prox_l1(l_penalty / eta,A @ beta + muk / eta)
        muk = muk + eta * (A @ beta - thetak)
        # objective value
        objective_admm_b_l1.append(objective_l1(beta, data, l_penalty))
        objective_admm.append(model.neg_log_like(beta, data) + l_penalty * np.linalg.norm(thetak,1))
        
        # if verbose:
        #     print("%d-th ADMM, objective value: %f"%(i+1, objective_admm_b_l1[-1]))
        # if abs(objective_admm_b_l1[-2] - objective_admm_b_l1[-1]) < ths:
        #     print("Converged!")
        #     break
        if verbose:
            out.write("%d-th ADMM, objective value: %f\n"%(i+1, objective_admm[-1]))
            out.flush()
        if objective_admm[-2] - objective_admm[-1] < ths:
            if verbose:
                out.write("Converged!\n")
                out.flush()
            break
        elif i >= max_iter - 1:
            if verbose:
                out.write("Not converged.\n")
                out.flush()
    
    beta = beta - sum(beta[0:N]) / N
    beta = beta.reshape((T,N))

    if return_b_obj:
        return (objective_admm, objective_admm_b_l1), beta
    return objective_admm, beta
