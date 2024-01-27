"""
script for running responsive setting experiments
"""

import numpy as np
import pandas as pd
import scipy
import random
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os
import argparse
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from scipy.stats import beta

# priors taken from EB prior of fixed model
PRIOR_A = 1.4270965891341989
PRIOR_B = 0.5916156493565227
PARAMS = [9.99983728e-01, 9.57030097e-01, 1.87562798e+00, 4.59026755e+00, 6.10832121e+00, 4.14816857e+00]
SCORES = [0,1,2,3,4,5]

class ProductHelper:
    """Class for helping keep track of all the products we have."""
    
    def __init__(self, universe, starting_market, mkt_ids, priors):
        self.universe = universe
        self.mkt_ids = mkt_ids
        self.priors = priors
        self.market = [np.array([copy.deepcopy(self.priors)[k]]) for k in starting_market]
        
    def pull_arm(self, action, rating):
        arr = self.market[action]
        latest_action = copy.deepcopy(arr[-1])
        array_to_add = np.zeros(len(PARAMS))
        array_to_add[rating] += 1
        latest_action += array_to_add
        self.market[action] = np.append(arr, [latest_action], axis=0)
        
    def pull_arm_update_market(self, action, rating):
        self.pull_arm(action, rating)
        for i in range(len(self.market)):
            if not np.array_equal(i, action):
                self.market[i] = np.append(self.market[i], [self.market[i][-1]], axis=0)
        
    def replace_item(self, old_id, new_id):
        old_idx = self.mkt_ids.index(old_id)
        self.universe[old_id].append(copy.deepcopy(self.market[old_idx]))
        self.mkt_ids[old_idx] = new_id
        self.market[old_idx] = np.array([copy.deepcopy(self.priors[new_id])])

def random_argmax(alist, rng=None):
    if not rng:
        rng = np.random.default_rng()
        
    maxval = max(alist)
    argmax = [idx for idx, val in enumerate(alist) if val == maxval]
    return rng.choice(argmax)

def ts_action(actions, param_array, rng=None):
    if not rng:
        rng = np.random.default_rng()
        
    p_hat = [np.dot(rng.dirichlet(ar, 1), SCORES) for ar in param_array]
    a = random_argmax(p_hat, rng)
    return a

def uniform_from_top_k(actions, param_array, rng=None, k=3):
    if not rng:
        rng = np.random.default_rng()

    est_quals = np.dot(param_array, SCORES)
    selections = np.argpartition(est_quals, -k)[-k:]
    a = random_argmax(selections, rng)
    return a
    

def sample_chosen_df(videos, chosen_df, action_index, id_columnname, review_columnname, rng=None):
    if not rng:
        rng = np.random.default_rng()
        
    vid = videos[action_index]
    seen_like = chosen_df[chosen_df[id_columnname]==vid].sample(1, random_state=rng).iloc[0][review_columnname]
    return seen_like

# core function for running simulation
def run_multiarmed_bandit_replenishment(chosen_df,
                                        videos,
                                        priors,
                                        id_columnname,
                                        review_columnname,
                                        sampling_action,
                                        timesteps,
                                        rho,
                                        mkt_size,
                                        num_users=1,
                                        snapshot_start=None,
                                        snapshotting_prob=0.001,
                                        seed=None,
                                        id_name=None):
    print(f'simulation {id_name} beginning...')
    rng=np.random.default_rng(seed)
    if not snapshot_start:
        snapshot_start = timesteps//5
    product_data = dict(zip(videos, [[] for _ in range(len(videos))]))
    priors_dict = dict(zip(videos, [priors.copy()[i,:] for i in range(priors.shape[0])]))
    snapshot_dict = dict()
    snapshot_num = 1
            
    curr_vids = np.array(list(rng.choice(videos, mkt_size, replace=False)))
    remaining_vids = set(videos).difference(set(curr_vids))

    helper = ProductHelper(product_data, curr_vids, list(curr_vids), priors_dict)
    market_history = []
    
    marker = 0

    for t in range(timesteps):
        if (t+1) % (timesteps//10) == 0:
            print(f'{t+1}/{timesteps}')
            
            
        market_history.append(copy.deepcopy(helper.mkt_ids))
        latest_sims = np.array([item[-1] for item in helper.market])
        actions = range(mkt_size)
        for m in range(num_users):
            a = sampling_action(actions, latest_sims, rng=None)
            chosen_action_global_index = videos.index(helper.mkt_ids[a])
            market_history[-1].append(copy.deepcopy(helper.mkt_ids[a]))
            like = sample_chosen_df(videos, 
                                    chosen_df, 
                                    chosen_action_global_index, 
                                    id_columnname, 
                                    review_columnname, 
                                    rng=None)

            # update prior
            helper.pull_arm_update_market(a, like)

        # replenish the indices
        flips = rng.binomial(1, rho, mkt_size)
        draws = rng.choice(list(remaining_vids) + 
                           [helper.mkt_ids[i] for i in range(len(helper.mkt_ids)) if flips[i]==1], mkt_size,replace=False)

        replenishments = flips * draws
        replaced = flips * helper.mkt_ids
        swapped_pairs = zip(list(replaced[replaced != 0].flatten()), list(replenishments[replenishments != 0].flatten()))
        replenishments = set(replenishments[replenishments != 0].flatten().astype(int))
        replaced = set(replaced[replaced != 0].flatten().astype(int))
        remaining_vids = remaining_vids.union(replaced).difference(replenishments)
        
        # ensure that the remaining videos are distinct size is constant
        if len(list(remaining_vids)) != len(videos) - mkt_size:
            print('remaining_vids', len(list(remaining_vids)))
            print('curr_vids', helper.mkt_ids)
            print('replenishments', replenishments)
            print('replaced', replaced)
            print('flips', flips)
            print('draws', draws)
            assert False

        for old,new in swapped_pairs:
            helper.replace_item(old, new)
            
        if t >= snapshot_start and rng.binomial(1, snapshotting_prob) > 0:
            snapshot_dict[snapshot_num] = (copy.deepcopy(helper.mkt_ids), copy.deepcopy(helper.market))
            snapshot_num += 1
            
    for prod in helper.mkt_ids:
        mkt_idx = helper.mkt_ids.index(prod)
        helper.universe[prod].append(helper.market[mkt_idx])
    
    #rename everything by id name if given
    if id_name:
        for k in list(helper.universe.keys())[:]:
            helper.universe[(k, id_name)] = helper.universe.pop(k)
            
        for k in list(snapshot_dict.keys())[:]:
            snapshot_dict[(k, id_name)] = snapshot_dict.pop(k)
            
    return helper.universe, snapshot_dict, np.array(market_history)

# commenting this out since it's not tuned for goodreads data yet
# def setup_data(num_samples=100):
#     # setup and transform the data
#     kuairec_test = pd.read_csv('kuairec_test.csv')
    
#     df = kuairec_test[['video_id', 'like_ratio']].drop_duplicates().sort_values(by='like_ratio', ascending=False)
#     test_videos = pd.Series(data=df['like_ratio'])
#     test_videos.index = df['video_id']
    
#     sampled_videos = (list(test_videos.sample(num_samples, random_state=1729).keys()))
#     kuairec_chosen = kuairec_test[kuairec_test['video_id'].isin(sampled_videos)]
    
#     return sampled_videos, kuairec_chosen

def upsample(datapath, num_samples, id_columnname, score_columnname, custom_percentiles=None):
    dataset = pd.read_csv(datapath)
    
    if not custom_percentiles:
        percentiles = np.linspace(0,100,num_samples+1) / 100
    else:
        percentiles = np.array(custom_percentiles) / 100
    
    df = dataset[[id_columnname, score_columnname]].drop_duplicates().sort_values(by=score_columnname)
    percentiles = np.round(percentiles * len(df)).astype(int)
    percentiles = percentiles[:-1] #100th percentile doesn't exist omit it
    sampled_videos = list(df.iloc[percentiles][id_columnname])
    chosen_df = dataset[dataset[id_columnname].isin(sampled_videos)]
    
    return sampled_videos, chosen_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate simulation outputs for online marketplace experiment.')
    parser.add_argument('--mode', choices=['test', 'full'], default='full')
    parser.add_argument('--timestep', type=int, default=5000000)
    parser.add_argument('--exit_rate', type=float, default=0.01)
    parser.add_argument('--market_size', type=int, default=5)
    parser.add_argument('--universe_size', type=int, default=25)
    parser.add_argument('--dataset',type=str, default='goodreads_robust_test.csv')
    parser.add_argument('--sampling', type=str, default='ts')
    args = parser.parse_args()
    
    # sampled_videos, kuairec_chosen = setup_data(args.universe_size)
    sampled_videos, kuairec_chosen = upsample(datapath=args.dataset, 
                                              num_samples=args.universe_size,
                                              id_columnname='book_id',
                                              score_columnname='quality')
    
    uninformed_priors = np.ones(len(sampled_videos)*len(PARAMS)).reshape(len(sampled_videos),len(PARAMS))
    eb_priors = np.tile(np.array(PARAMS), (len(sampled_videos), 1))
    
    folder_name = 'raw_responsive_data'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    etas_to_test = [0.001,1,10,50,100,500,1000]
    
    prior_settings = []
    prior_names = []
    for a in np.array(etas_to_test).astype(float):
        curr_prior = a*eb_priors
        prior_settings.append(curr_prior)
        prior_names.append(np.round(a, 2))

    sampling_algo = None
    if args.sampling=='top_k':
        sampling_algo = uniform_from_top_k
    else:
        sampling_algo=ts_action
        
    if args.mode=='test':
        data, snapshots, market_histories = run_multiarmed_bandit_replenishment(kuairec_chosen,
                                                              sampled_videos,
                                                              prior_settings[0],
                                                              "book_id",
                                                              "review",
                                                              sampling_algo,
                                                              timesteps=args.timestep,
                                                              rho=args.exit_rate,
                                                              mkt_size=args.market_size,
                                                              seed=1729,
                                                              id_name = prior_names[0])
        np.save(f'{folder_name}/sim_data_alpha_{prior_names[0]}.npy', data)
        np.save(f'{folder_name}/sim_snapshots_alpha_{prior_names[0]}.npy', snapshots)
        np.save(f'{folder_name}/market_id_data_{prior_names[0]}.npy', market_histories)
    else:
        parallel = Parallel(n_jobs=len(etas_to_test), verbose=10)
        result_data = parallel(delayed(run_multiarmed_bandit_replenishment)(kuairec_chosen,
                                                              sampled_videos,
                                                              prior_settings[i],
                                                              "book_id",
                                                              "review",
                                                              sampling_algo,
                                                              timesteps=args.timestep,
                                                              rho=args.exit_rate,
                                                              mkt_size=args.market_size,
                                                              seed=1729,
                                                              id_name = prior_names[i]) \
                                                              for i in range(len(prior_names)))
        
        result_data = list(result_data)
        print(len(result_data))
        for i in range(len(result_data)):
            data, snapshots, market_histories = result_data[i]
            np.save(f'{folder_name}/sim_data_alpha_{prior_names[i]}.npy', data)
            np.save(f'{folder_name}/sim_snapshots_alpha_{prior_names[i]}.npy', snapshots)
            np.save(f'{folder_name}/market_id_data_{prior_names[i]}.npy', market_histories)
    
    
    print(f'saved results for params={PARAMS} to {folder_name} folder.')
        