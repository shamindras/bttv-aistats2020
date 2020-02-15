import numpy as np
import scipy as sc
import scipy.linalg as spl
import scipy.stats as ss
import pandas as pd
import sys, os, csv

import grad_utils as model
import cv_utils
import opt_utils
import ks_utils as ks


def max_change(beta):
    '''
    get the maximal change in rank in neighboring timepoint based on beta
    '''
    T,N = beta.shape
    arg = np.array([ss.rankdata(-beta[ii]) for ii in range(T)])
    return np.max(abs(arg[1:] - arg[:-1]))

def plot_nfl_round(beta, team_id,season):
    T, N = beta.shape
    year = range(1,17)
    f = plt.figure(1, figsize = (6,4))

    for i in range(N):
        plt.plot(year,beta[:,i], label=team_id['name'][i], color = np.random.rand(3,))
    plt.xlabel("round")
    plt.ylabel("latent parameter")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1, 1, 0),prop={'size': 5})
    plt.ticklabel_format(style='plain',axis='x',useOffset=False)

    f.savefig("nfl_round_"+str(season)+".pdf", bbox_inches='tight')
        
def get_elo_rank_season(elo_all, season):
    elo_season = elo_all.iloc[np.where(elo_all['season'] == season)]
    elo_season = elo_season[pd.isnull(elo_season['playoff'])]
    a = elo_season[['team1','elo1_post']]
    a.columns = ['team','elo']
    a = a.reset_index()
    b = elo_season[['team2','elo2_post']]
    b.columns = ['team','elo']
    b = b.reset_index()

    c = pd.concat([a,b])
    c = c.sort_values(by = ['index'])    
    d = c.groupby(by = ['team']).last()
    
    x = d.index.values
    x[np.where(x == 'LAR')] = 'STL'
    x[np.where(x == 'LAC')] = 'SD'
    x[np.where(x == 'JAX')] = 'JAC'
    x[np.where(x == 'WSH')] = 'WAS'
    
    elo_rank = pd.DataFrame({'rank': ss.rankdata(-d['elo'])},index = x).sort_index()
    
    return elo_rank

def get_single_round_pwise(rnd_num, nfl_data_dir, season):
    """
    Gets the pairwise numpy array of score diffences across teams for a single
       round in a season
    """
    fname = "round" + "_" + str(rnd_num).zfill(2) + ".csv"
    fpath = os.path.join(nfl_data_dir, str(season), fname)
    rnd_df = pd.read_csv(fpath)
    pwise_diff = rnd_df.pivot(index='team', columns='team_other',values='diff').values
    pwise_diff[pwise_diff >= 0] = 1
    pwise_diff[pwise_diff < 0] = 0
    pwise_diff[np.isnan(pwise_diff)] = 0
    return pwise_diff

def get_final_rank_season(data_dir, season, team_id, all_rnds, plot = True, 
                          loocv= True, threshold = 3,num_loocv = 200):
    game_matrix_list = np.array([get_single_round_pwise(rnd_num=rnd, nfl_data_dir=data_dir, season=season) 
                                  for rnd in all_rnds])
    
    T, N = game_matrix_list.shape[:2]
    if loocv:
        h_list = np.linspace(0.5, 0.05, 15)
        h_star, nll_cv, beta = cv_utils.loocv_ks(game_matrix_list, h_list,
                                                 opt_utils.gd_bt, num_loocv = num_loocv, return_prob = False,
                                                 verbose='cv', out='notebook')    
    else:
        h_list = np.linspace(0.5, 0.05, 15)
        val_list = []

        data = game_matrix_list
        for i in range(len(h_list)):
            h = h_list[i]
            ks_data = kernel_smooth(data,h)
            val_list.append(max_change(_, beta = opt_utils.gd_bt(ks_data,l_penalty = 0)))

        # plt.plot(lam_list,val_list)

        while val_list[-1] > threshold:
            threshold += 1

        ix = next(idx for idx, value in enumerate(val_list) if value <= threshold)
        h_star = h_list[ix]
        
        ks_data = kernel_smooth(data,h_star)
        beta = opt_utils.pgd_l2_sq(ks_data,l_penalty = 0)

    if plot:
        plot_nfl_round(beta = beta,team_id = team_id,season = SEASON)

    arg = np.argsort(-beta,axis=1)
    rank_list = pd.DataFrame(data={(i):team_id['name'][arg[i-1,]].values for i in range(1,2)})
    rank_last = rank_list[1]
    rank_last = pd.DataFrame({'rank':range(len(rank_last))},index = rank_last.values)
    
    print("season " + str(season) + " finished. \n")
    
    return rank_last.sort_index() + 1



def get_team_season(season_idx, model_season_list):
    model_season = model_season_list[season_idx]
    model_season['rank'] = model_season['rank'].astype(np.int64)
    model_season['team'] = model_season.index
    model_season = model_season.sort_values(by='rank', ascending=True)
    return model_season

def get_join_elo_bt_season(season_num, elo_team_season, bt_team_season, top_n):
    top_season = elo_team_season.merge(bt_team_season, how='left', on=['rank'])
    top_season.columns = ["rank", f"ELO {season_num}", f"BT {season_num}"]
    top_season = top_season[[f"ELO {season_num}", f"BT {season_num}"]].head(top_n)
    #top_season = top_season.reset_index(drop=True)
    return top_season