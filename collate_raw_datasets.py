import numpy as np
import pandas as pd
import scipy
import random
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import itertools
import os
import json
import argparse

import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from scipy.stats import beta

RAW_DATASET_NAMES = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collate datasets for online marketplace experiment.')
    parser.add_argument('-ds','--datasets', nargs='+', help='<Required> Set flag', required=True)
    args = parser.parse_args()
    
    #collate the raw data together and make a dataframe with them
    for ds_path in args.datasets:
        with open(ds_path, 'r') as file:
            datalines = file.readlines()
            datadict = []
            file.close()
            for line in datalines:
                review = json.loads(line)
                review.pop("review_text", None)
                datadict.append(review)
    
    goodreads_df = pd.DataFrame(datadict)
    goodreads_df = goodreads_df.drop_duplicates(subset="review_id")
    
    # do the train test split and calculate average quality
    book_df = goodreads_df[['book_id', 'rating']].groupby(['book_id']).mean()['rating']
    train_df, test_df = train_test_split(book_df, test_size = 0.4, random_state=1729)
    goodreads_test = pd.merge(goodreads_df, test_df, on='book_id')
    goodreads_train = pd.merge(goodreads_df, train_df, on='book_id')
    goodreads_test = goodreads_test.rename(columns={"rating_x": "review", "rating_y": "quality"})
    goodreads_train = goodreads_train.rename(columns={"rating_x": "review", "rating_y": "quality"})

    # sanity check by printing out head of test dataset
    print(goodreads_test.head())

