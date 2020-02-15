import pandas as pd
import scipy as sc
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags


def get_winrate(data):
    T, N = data.shape[:2]
    winrate = np.sum(data, 2) / (np.sum(data,2) + np.sum(data,1))
    return winrate

def rank_list(v):
    output = [0] * len(v)
    for i, x in enumerate(sorted(range(len(v)), key=lambda y: v[y])):
        output[x] = i
    return np.array(output)

def av_dif_rank(beta1,beta2):
    # both beta should be T-by-N matrices
    T, N = beta1.shape
    result = [0] * T
    for t in range(T):
        beta1t = beta1[t]
        beta2t = beta2[t]
        rank1t = rank_list(beta1t)
        rank2t = rank_list(beta2t)
        result[t] = np.mean(abs(rank1t - rank2t))
    return result


def beta_gaussian_process(N, T, mu_parameters, cov_parameters, mu_type = 'constant', cov_type = 'toeplitz'):
    '''
    generate beta via a Gaussian process
    '''
    if mu_type == 'constant':
        loc, scale = mu_parameters
        mu_start = stats.norm.rvs(loc = loc,scale = scale,size = N,random_state = 100)
        mu = [np.ones(T) * mu_start[i] for i in range(N)]
    if cov_type == 'toeplitz':
        alpha, r = cov_parameters
    ##### strong auto-correlation case, off diagonal  = 1 - T^(-alpha) * |i - j|^r
    off_diag = 1 - T ** (-alpha) * np.arange(1,T + 1) ** r
    cov_single_path = sc.linalg.toeplitz(off_diag,off_diag)

    return np.array([np.random.multivariate_normal(mean = mu[i],cov = cov_single_path,size = 1).ravel() for i in range(N)]).T


def get_game_matrix_list(N,T,tn,beta):
    '''
    get the list of T game matrices
    -------------
    Input:
    N: number of teams
    T: number of seasons
    tn: list of number of games between each pair of teams
    beta: a T-by-N array
    -------------
    Output:
    game_matrix_list: a 3-d np.array of T game matrices, each matrix (t,:,:) stores the number of times that i wins j at entry (i,j) at season t
    '''
    game_matrix_list = [None] * T
    ind = -1 # a counter used to get tn
    for t in range(T):
        game_matrix = np.zeros(N * N).reshape(N,N)
        for i in range(N):
            for j in range(i + 1,N):
                ind += 1
                pij = np.exp(beta[t,i] - beta[t,j]) / (1 + np.exp(beta[t,i] - beta[t,j]))
                nij = np.random.binomial(n = tn[ind], p = pij, size = 1)
                game_matrix[i,j],game_matrix[j,i] = nij, tn[ind] - nij
        game_matrix_list[t] = game_matrix
    return np.array(game_matrix_list)

def beta_markov_chain(num_team,num_season,var_latent = 1,coef_latent = 1,sig_latent = 1,draw = True):
    pc_latent = diags([-coef_latent/var_latent, 1/var_latent, -coef_latent/var_latent],
                      offsets = [-1, 0, 1],
                      shape=(num_season, num_season)).todense()
    pc_latent[np.arange(num_season-1), np.arange(num_season-1)] += (coef_latent**2)/var_latent
    var_latent = np.linalg.inv(pc_latent)
    inv_sqrt_var = np.diag(1/np.sqrt(np.diag(var_latent)))
    var_latent = sig_latent * inv_sqrt_var @ var_latent @ inv_sqrt_var
    
    latent = np.transpose(
        np.random.multivariate_normal(
            [0]*num_season, var_latent, num_team))
    
    if draw:
        f = plt.figure(1, figsize = (12,5))
        
        ax = plt.subplot(121)
        plt.imshow(var_latent, 
           cmap='RdBu', vmin=-sig_latent, vmax=sig_latent)
        plt.colorbar()
        plt.title("variance")
        
        ax = plt.subplot(122)
        plt.imshow(latent, cmap='RdBu',
           vmin=-np.max(np.abs(latent)), 
           vmax=np.max(np.abs(latent)))
        plt.xlabel("team number")
        plt.ylabel("season number")
        plt.title("beta")
    return latent



################### for agnostic setting #######################

def get_prob_matrix(beta):
    # get the probability matrix given the value of beta
    # beta should be a T-by-N matrix
    T, N = beta.shape
    # l stores the loglikelihood
    N_one = np.ones(N).reshape(N,1)
    P = [None] * T
    for t in range(T):
        b = beta[t,:].reshape(N,1)
        D = b @ N_one.T - N_one @ b.T
        W = (1 + np.exp(D))
        P_this = 1 / W.T
        np.fill_diagonal(P_this,0)
        P[t] = P_this
    return np.array(P)

def path_gaussian_process(N, T, mu_parameters, cov_parameters, mu_type = 'constant', cov_type = 'toeplitz'):
    '''
    generate N paths of a Gaussian process
    '''
    spread = mu_parameters[1] - mu_parameters[0]
    mu_start = []
    if mu_type == 'constant':
#         loc, scale = mu_parameters
#         mu_start = stats.norm.rvs(loc = loc,scale = scale,size = N,random_state = 100)
        n_group = 5
        group_id = np.array_split(range(N), n_group)
        for i in range(n_group):
            mu_start += list(np.random.uniform(low=mu_parameters[0] + 3 * spread * i, 
                                          high=mu_parameters[1] + 3 * spread * i, size=len(group_id[i])))
        mu = [np.ones(T) * mu_start[i] for i in range(N)]
    if cov_type == 'toeplitz':
        alpha, r = cov_parameters
    ##### strong auto-correlation case, off diagonal  = 1 - T^(-alpha) * |i - j|^r
    off_diag = 1 - T ** (-alpha) * np.arange(1,T + 1) ** r
    cov_single_path = sc.linalg.toeplitz(off_diag,off_diag)

    return np.array([np.random.multivariate_normal(mean = mu[i],cov = cov_single_path,size = 1).ravel() for i in range(N)]).T


def make_prob_matrix(T,N,alpha = 0.5,r = 0.5,mu = [0,0.5]):
    # get the probability matrix given the value of beta
    # beta should be a T-by-N matrix
#     P = [None] * T
#     for t in range(T):
#         P_this = np.triu(np.random.uniform(low=0.1, high=0.9, size=N * N).reshape((N,N)),1)
#         P_this = np.triu(1-P_this).T + P_this
#         np.fill_diagonal(P_this,0)
#         P[t] = P_this
    P_entry = path_gaussian_process(int(N * N), T, mu_parameters = mu, cov_parameters = [alpha,r], mu_type = 'constant', cov_type = 'toeplitz')
    order = np.argsort(P_entry[:,0])
    P_entry = P_entry[order,:]
    index = np.random.permutation(N)
    
    p_low = 0.05
    p_up = 0.95
    p_min = min(P_entry.flatten())
    p_max = max(P_entry.flatten())

    P_entry = (P_entry - p_min) * (p_up - p_low)/(p_max - p_min) + p_low

    P = [None] * T
    for t in range(T):
        P_this = np.triu(P_entry[t].reshape((N,N)),1)
        P_this = np.triu(1-P_this).T + P_this
        np.fill_diagonal(P_this,0)
        P[t] = P_this[np.ix_(index,index)]    
    return np.array(P)


def get_game_matrix_list_from_P(tn,P_list):
    '''
    get the list of T game matrices
    -------------
    Input:
    tn: list of number of games between each pair of teams
    P_list: a T-by-N-by-N array
    -------------
    Output:
    game_matrix_list: a 3-d np.array of T game matrices, each matrix (t,:,:) stores the number of times that i wins j at entry (i,j) at season t
    '''
    T, N = P_list.shape[:-1]
    game_matrix_list = [None] * T
    ind = -1 # a counter used to get tn
    for t in range(T):
        game_matrix = np.zeros(N * N).reshape(N,N)
        for i in range(N):
            for j in range(i + 1,N):
                ind += 1
                pij = P_list[t,i,j]
                nij = np.random.binomial(n = tn[ind], p = pij, size = 1)
                game_matrix[i,j],game_matrix[j,i] = nij, tn[ind] - nij
        game_matrix_list[t] = game_matrix
    return np.array(game_matrix_list)













'''
some examles of running functions beta_gaussian_process and get_game_matrix_list
##### example of generating

N = 10 # number of teams
T = 10 # number of seasons/rounds/years
tn_median = 100
bound_games = [tn_median - 2,tn_median + 2] # bounds for the number of games between each pair of teams

##### tn: list of number of games between each pair of teams
tn = stats.randint.rvs(low = int(bound_games[0]), high = int(bound_games[1]), size = int(T * N * (N - 1) / 2))

##### get beta here #####
beta = beta_gaussian_process(N, T, mu_parameters = [0,1], cov_parameters = [alpha,r], mu_type = 'constant', cov_type = 'toeplitz')

game_matrix_list = get_game_matrix_list(N,T,tn,beta)


##### example of drawing paths

beta = beta_gaussian_process(N, T, mu_parameters = [beta_mu], cov_parameters = [alpha,r], mu_type = 'constant', cov_type = 'toeplitz')

f = plt.figure(3, figsize = (9,4.5))
ax = plt.subplot(111)
for i in range(T):
    ax.plot(range(1,T + 1),beta[i],c=np.random.rand(3,1),marker = '.',label = 'Team' + str(i),linewidth=1)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
'''

