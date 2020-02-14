import numpy as np

def neg_log_like(beta,game_matrix_list):
    '''
    compute the negative loglikelihood
    ------------
    Input:
    
    beta: can be a T-by-N array or a T*N-by-1 array
    game_matrix_list: records of games, T-by-N-by-N array
    ------------
    -l: negative loglikelihood, a number
    '''
    # beta could be a T-by-N matrix or T*N-by-1 array
    T, N = game_matrix_list.shape[0:2]
    beta = beta.reshape(T,N)
    # l stores the loglikelihood
    l = 0
    N_one = np.ones(N).reshape(N,1)
    for t in range(T):
        Cu = np.triu(game_matrix_list[t]) # equivalent to [t,:,:]
        Cl = np.tril(game_matrix_list[t])
        b = beta[t,:].reshape(N,1)
        D = b @ N_one.T - N_one @ b.T
        W = np.log(1 + np.exp(D))
        l += N_one.T @ (Cu * D) @ N_one - N_one.T @ ((Cu + Cl.T) * np.triu(W)) @ N_one
    return -l[0,0]


def grad_nl(beta,game_matrix_list):
    '''
    compute the gradient of the negative loglikelihood
    ------------
    Input:
    beta: can be a T-by-N array or a T*N-by-1 array
    game_matrix_list: records of games, T-by-N-by-N array
    ------------
    Output:
    -grad: gradient of negative loglikelihood, a T*N-by-1 array
    '''
    # beta could be a T-by-N array or a T*N-by-1 array
    T, N = game_matrix_list.shape[0:2]
    beta = beta.reshape(T,N)
    # g stores the gradient
    g = np.zeros(N * T).reshape(T,N)
    N_one = np.ones(N).reshape(N,1)
    for t in range(T):
        C = game_matrix_list[t]
        b = beta[t,:].reshape(N,1)
        W = np.exp(b @ N_one.T) + np.exp(N_one @ b.T)
        g[t,:] = ((C / W) @ np.exp(b) - (C / W).T @ N_one * np.exp(b)).ravel()
    return -g.reshape(N * T,1)

def hess_nl(beta,game_matrix_list):
    '''
    compute the Hessian of the negative loglikelihood
    ------------
    Input:
    beta: can be a T-by-N array or a T*N-by-1 array
    game_matrix_list: records of games, T-by-N-by-N array
    ------------
    Output:
    -H: Hessian of negative loglikelihood T*N-by-T*N array
    '''
    # beta could be a T-by-N array or a T*N-by-1 array
    T, N = game_matrix_list.shape[0:2]
    beta = beta.reshape(T,N)
    # H stores the Hessian
    H = np.zeros(N ** 2 * T ** 2).reshape(T * N,T * N)
    N_one = np.ones(N).reshape(N,1)
    for t in range(T):
        Cu = np.triu(game_matrix_list[t]) # equivalent to [t,:,:]
        Cl = np.tril(game_matrix_list[t])
        Tm = Cu + Cl.T + Cu.T + Cl
        b = beta[t,:].reshape(N,1)
        W = np.exp(b @ N_one.T) + np.exp(N_one @ b.T)
        H_t = Tm * np.exp(b @ N_one.T + N_one @ b.T) / W ** 2
        H_t += -np.diag(sum(H_t))
        ind = range(t * N, (t + 1) * N)
        H[t * N:(t + 1) * N,t * N:(t + 1) * N] = H_t
    return -H




'''
##### example

N = 10
T = 10
tn_median = 100
bound_games = [tn_median - 2,tn_median + 2] # bounds for the number of games between each pair of teams

##### tn: list of number of games between each pair of teams
tn = stats.randint.rvs(low = int(bound_games[0]), high = int(bound_games[1]), size = int(T * N * (N - 1) / 2))

beta0 = beta_gaussian_process(N, T, mu_parameters = [0,1], cov_parameters = [alpha,r], mu_type = 'constant', cov_type = 'toeplitz')
game_matrix_list = get_game_matrix_list(N,T,tn,beta0)


loglike(beta0,game_matrix_list)
g = grad_l(beta0,game_matrix_list)
H = Hess_l(beta0,game_matrix_list)





##### test case (numerically)

# verify gradient
delta = 0.0001
g_hat = np.zeros((N * T,1))

for i in range(N * T):
    beta1 = beta0.copy().reshape(N * T,1).ravel() 
    beta2 = beta1.copy() 
    beta1[i] -= delta
    beta2[i] += delta
    g_hat[i] = (loglike(beta2,game_matrix_list) - loglike(beta1,game_matrix_list)) / (2 * delta)

print(sum(abs(g - g_hat)))


# verify Hessian
delta = 0.0001
H_hat = np.zeros((N * T,N * T))

for i in range(N * T):
    beta1 = beta0.copy().reshape(N * T,1).ravel() 
    beta2 = beta1.copy() 
    beta1[i] -= delta
    beta2[i] += delta
    H_hat[i] = (grad_l(beta2,game_matrix_list) - grad_l(beta1,game_matrix_list)).ravel() / (2 * delta)

print(sum(sum(abs(H_hat - H))))

'''