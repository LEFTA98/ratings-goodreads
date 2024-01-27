import numpy as np
import pandas as pd
import scipy
import random
import matplotlib.pyplot as plt
import seaborn as sns
import copy

import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from scipy.stats import beta
import dirichlet

MIN_NUM_REVIEWS = 10

if __name__ == "__main__":
    # filter train dataset
    df = pd.read_csv('goodreads_train.csv')
    df = df[['book_id', 'review', 'date_added']].pivot_table(index='book_id', columns='review', aggfunc=lambda x: len(x.unique()))
    df = df.fillna(0)
    df = df.reset_index()
    df.columns = ['book_id', 'num_zero', 'num_one', 'num_two', 'num_three', 'num_four', 'num_five']
    df['n'] = df.drop('book_id', axis=1).sum(axis=1)
    df = df[(df['n'] > MIN_NUM_REVIEWS) & df.drop(['book_id', 'n'], axis=1).product(axis=1) > 0]
    
    # display the parameters
    params = dirichlet.mle(np.array(df.drop(['book_id', 'n'],axis=1))/np.tile(df['n'],(6,1)).T, method='fixedpoint')
    print('dirichlet prior parameters are:')
    print(params)
    
    # save a kernel density estimate plot of the prior dstbn
    r = scipy.stats.dirichlet.rvs(params, size=5000)
    f = sns.kdeplot(r @ np.array([0,1,2,3,4,5])).set_title('density plot of expected quality, eb prior').get_figure()
    f.savefig('prior_kde.png')
    
    # filter test dataset and save it as a "clean" dataset for the responsive experiments
    df_test = pd.read_csv('goodreads_test.csv')[['user_id', 'book_id', 'review_id', 'review', 'date_added', 'quality']]
    df_test['date_added'] = pd.to_datetime(df_test['date_added'].str[:20] + df_test['date_added'].str[26:])
    df_test = df_test.sort_values(by='date_added')
    df_test['rank'] = df_test.groupby('book_id')['date_added'].rank()
    df_test = pd.merge(df_test, df_test.groupby('book_id')['rank'].count(), on='book_id')
    df_test = df_test.rename(columns={'rank_x': 'rank', 'rank_y': 'max_rank'})
    df_test = df_test[(df_test['max_rank'] >= 100) & (df_test['rank'] <= 100)]
    df_test.to_csv("goodreads_robust_test.csv")
    
    df_calibrate = df_test[df_test['rank'] <= 50]
    df_eval = df_test
    test_df_videos = set(df_test['book_id'])
    
    ###
    # params to use for fixed experiments
    ###
    prior_set = np.array([
                            [0,0,0,0,0,0],
                             params,
                             params * 10,
                             params * 1000])
    etas = np.array([0,1,10,1000])
    
    print('parameterizations to test')
    print(prior_set)
    calibration_df = {'params': [], 'eta': [], 'num_samples': [], 'book_id': []}
    max_calibration_samples = 50
    
    for i in range(len(etas)):
        print(f'calibrating eta value {etas[i]}')
        for b_id in set(df_test['book_id']):
            for j in range(1,max_calibration_samples+1):
                used_results = df_calibrate[(df_calibrate['book_id']==b_id) 
                                                 & (df_calibrate['rank'] <= j)]
                base_params = prior_set[i]
                for result in used_results['review']:
                    base_params[result] += 1
                calibration_df['params'].append(copy.deepcopy(base_params))
                calibration_df['eta'].append(etas[i])
                calibration_df['num_samples'].append(j)
                calibration_df['book_id'].append(b_id)
                
                
    calibration_df = pd.DataFrame(calibration_df)
    params_np = np.stack(calibration_df['params'].to_numpy())
    params_np = np.stack(calibration_df['params'].to_numpy())
    calibration_df['est_quality'] = (params_np @ np.array([0,1,2,3,4,5]))/(np.sum(params_np, axis=1))
    
    evaluation_df = df_eval.pivot_table(index='book_id', values='user_id', columns='review', aggfunc='count').fillna(0)
    evaluation_df.columns = [0,1,2,3,4,5]
    evaluation_df['true_quality'] = np.array(evaluation_df) @ np.array([0,1,2,3,4,5]) / evaluation_df.sum(axis=1)
    
    results_df = pd.merge(calibration_df, evaluation_df, on='book_id')
    results_df['eta'] = np.round(results_df['eta'],1)
    results_df['mse'] = np.square(results_df['true_quality'] - results_df['est_quality'])
    
    results_df.to_csv('fixed_results_df.csv')
    print("results saved from fixed experiments.")
