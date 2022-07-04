#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import json
import os
import numpy as np
import pandas as pd
import arff
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
import shutil
from helpers.utils import create_folder


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'count']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def print_density_level(data_dir='data/'):

    tp = pd.read_csv(data_dir + 'tp.csv')
    usercount, songcount = get_count(tp, 'uid'), get_count(tp, 'sid')
    density_level = float(tp.shape[0]) / (usercount.shape[0] * songcount.shape[0])
    print("After filtering, there are %d triplets from %d users and %d songs - density: %.3f%%"
          % (tp.shape[0], usercount.shape[0], songcount.shape[0], density_level*100))

    return density_level


def load_filter_record_tp(data_dir='data/', min_uc=20, min_sc=50, min_c=5):

    # Load Taste Profile data
    TP_file = os.path.join(data_dir, 'train_triplets.txt')
    tp = pd.read_table(TP_file, header=None, names=['uid', 'sid', 'count'])

    # Load the list of song IDs in the dataset,
    msd_track_file = os.path.join(data_dir, 'unique_tracks.txt')
    list_sids = list(pd.read_table(msd_track_file, header=None, sep='<SEP>', usecols=[1], engine='python')[1])

    # Keep the TP data whose sid are in the list of unique tracks
    tp = tp[tp['sid'].isin(list_sids)]

    # Only keep ratings >= min_c, otherwise they're considered not reliable.
    tp = tp[tp['count'] >= min_c]

    # Only keep the triplets for songs which were listened to at least 'min_sc' times
    songcount = get_count(tp, 'sid')
    valid_songs = songcount['sid'][songcount.loc[:, "size"] >= min_sc]
    tp = tp[tp['sid'].isin(valid_songs)]

    # Only keep the triplets for users who listened to at min_uc songs
    usercount = get_count(tp, 'uid')
    valid_users = usercount['uid'][usercount.loc[:, "size"] >= min_uc]
    tp = tp[tp['uid'].isin(valid_users)]

    # Record TP
    tp.to_csv(data_dir + 'tp.csv', index=False)

    return


def load_record_features(data_dir='data/'):

    # List of unique songs in the filtered TP data, if not provided as an input
    unique_sid = list(pd.unique(pd.read_csv(data_dir + 'tp.csv')['sid']))

    # Load the list of song IDs in the dataset and corresponding track IDs for mapping the two
    msd_track_file = os.path.join(data_dir, 'unique_tracks.txt')
    tid_sid = pd.read_table(msd_track_file, header=None, sep='<SEP>', usecols=[0, 1], engine='python')
    tid_sid.columns = ['tid', 'sid']
    tid_sid.set_index('tid', inplace=True)

    features_full = np.empty((168, 0))
    list_sid = []

    feat_loader = arff.load(data_dir + 'msd-ssd-v1.0.arff')
    for ii, row in enumerate(feat_loader):
        current_tid = row._values[-1]
        current_sid = tid_sid.loc[current_tid][0]
        if (current_sid in unique_sid) and not(current_sid in list_sid):
            list_sid.append(current_sid)
            features_full = np.concatenate((features_full, np.array(row._values[:-1])[:, np.newaxis]), axis=1)

    # Scale the features and store them into a pandas frame
    features_full = features_full.T
    features_full = scale(features_full, axis=0)
    features_pd = pd.DataFrame(features_full)
    features_pd.insert(0, 'sid', list_sid)

    # Record the list of features
    features_pd.to_csv(data_dir + 'features.csv', index=False)

    return features_pd


def update_tp_record(data_dir='data/', seed=12345):

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Load the TP data
    tp = pd.read_csv(data_dir + 'tp.csv')

    # Get the list of songs ids for which there are features
    unique_sid = pd.unique(pd.read_csv(data_dir + 'features.csv')['sid'])
    n_songs = len(unique_sid)

    # Update the TP songs
    tp = tp[tp['sid'].isin(unique_sid)]
    # Record updated TP
    tp.to_csv(data_dir + 'tp.csv', index=False)

    # Get users in TP
    unique_uid = pd.unique(tp['uid'])

    # Shuffle songs
    idx = np.random.permutation(np.arange(n_songs))
    unique_sid = unique_sid[idx]

    # Create a dictionary for mapping user/song unique ids to integers
    user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
    song2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))

    # Record TP data, dictionaries and lists of unique sid/uid
    with open(data_dir + 'unique_uid.txt', 'w') as f:
        for uid in unique_uid:
            f.write('%s\n' % uid)

    with open(data_dir + 'unique_sid.txt', 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    with open(data_dir + 'user2id.json', 'w') as f:
        json.dump(user2id, f)

    with open(data_dir + 'song2id.json', 'w') as f:
        json.dump(song2id, f)

    return


def split_cold(data_dir='data/', n_splits=10, seed=125):

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Load Taste profile data and the SSDs features
    tp = pd.read_csv(data_dir + 'tp.csv')
    my_features = pd.read_csv(data_dir + 'features.csv')

    # Load the list of unique songs
    unique_sid = []
    with open(data_dir + 'unique_sid.txt', 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    #np.random.shuffle(unique_sid)

    # Select 20% of the songs as held-out validation data
    n_songs = len(unique_sid)
    val_sid = unique_sid[:int(0.2 * n_songs)]

    # Get the corresponding validation TP data and features
    val_tp = tp[tp['sid'].isin(val_sid)]
    val_feats = my_features[my_features['sid'].isin(val_sid)]

    # Get the remaining songs IDs
    train_test_sid = np.array(unique_sid[int(0.2 * n_songs):])

    # Perform K splits on the remaining data (train and test)
    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(train_test_sid)
    for i_split, (train_index, test_index) in enumerate(kf.split(train_test_sid)):
        # Songs IDs
        train_sid = train_test_sid[train_index]
        test_sid = train_test_sid[test_index]

        # Corresponding TP data
        train_tp = tp[tp['sid'].isin(train_sid)]
        test_tp = tp[tp['sid'].isin(test_sid)]

        # Corresponding features
        train_feats = my_features[my_features['sid'].isin(train_sid)]
        test_feats = my_features[my_features['sid'].isin(test_sid)]

        # Define the current split folder
        current_path = data_dir + 'cold/split' + str(i_split) + '/'
        create_folder(current_path)

        # Save the .csv (TP data and features)
        train_tp.to_csv(current_path + 'train_tp.csv', index=False)
        val_tp.to_csv(current_path + 'val_tp.csv', index=False)
        test_tp.to_csv(current_path + 'test_tp.csv', index=False)

        train_feats.to_csv(current_path + 'train_feats.csv', index=False)
        val_feats.to_csv(current_path + 'val_feats.csv', index=False)
        test_feats.to_csv(current_path + 'test_feats.csv', index=False)

        # Save a copy of the list of unique sid/uid in each directory for convenience
        shutil.copyfile(data_dir + 'unique_sid.txt', current_path + 'unique_sid.txt')
        shutil.copyfile(data_dir + 'unique_uid.txt', current_path + 'unique_uid.txt')

    return


def numerize_cold(data_dir='data/', n_splits=10):

    # Load the user and song to id mappings
    with open(data_dir + 'user2id.json', 'r') as f:
        user2id = json.load(f)
    with open(data_dir + 'song2id.json', 'r') as f:
        song2id = json.load(f)

    # Numerize all the subsets / splits
    setting = 'cold'
    for i_split in range(n_splits):
        for subset_to_numerize in ['train', 'test', 'val']:
            current_path = data_dir + setting + '/split' + str(i_split) + '/' + subset_to_numerize

            # TP data
            data_tp = pd.read_csv(current_path + '_tp.csv')
            uid = list(map(lambda x: user2id[x], data_tp['uid']))
            sid = list(map(lambda x: song2id[x], data_tp['sid']))
            data_tp['uid'] = uid
            data_tp['sid'] = sid
            data_tp.to_csv(current_path + '_tp.num.csv', index=False)

            # Features
            data_feats = pd.read_csv(current_path + '_feats.csv')
            sid = list(map(lambda x: song2id[x], data_feats['sid']))
            data_feats = data_feats.assign(sid=sid)
            data_feats.to_csv(current_path + '_feats.num.csv', index=False)

    return


def split_warm(data_dir='data/', n_splits=10, seed=12345):

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Load Taste profile data
    tp = pd.read_csv(data_dir + 'tp.csv')

    # Load the list of unique songs
    unique_sid = []
    with open(data_dir + '/unique_sid.txt', 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    # Pick out 20% of the binarized playcounts for warm-start validation
    n_playcounts = tp.shape[0]
    vval = np.random.choice(n_playcounts, size=int(0.2 * n_playcounts), replace=False)
    val_idx = np.zeros(n_playcounts, dtype=bool)
    val_idx[vval] = True
    val_tp = tp[val_idx]

    # Get the remaining data for train/test, as well as the corresponding indices
    train_test_tp = tp[~val_idx]
    train_test_idx = np.array(train_test_tp.index.values.tolist())
    np.random.shuffle(train_test_idx)

    # Perform K splits on the remaining data (train and test)
    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(train_test_idx)

    for i_split, (train_index, test_index) in enumerate(kf.split(train_test_idx)):

        # Indices
        train_idx = train_test_idx[train_index]
        test_idx = train_test_idx[test_index]

        # Corresponding TP data
        train_tp = train_test_tp.loc[train_idx]
        test_tp = train_test_tp.loc[test_idx]

        # Define the current split folder
        current_path = data_dir + 'warm/split' + str(i_split) + '/'
        create_folder(current_path)

        # Save the .csv (TP data and features)
        train_tp.to_csv(current_path + 'train_tp.csv', index=False)
        val_tp.to_csv(current_path + 'val_tp.csv', index=False)
        test_tp.to_csv(current_path + 'test_tp.csv', index=False)

        # Save a copy of the list of unique sid/uid in each directory for convenience
        shutil.copyfile(data_dir + 'unique_sid.txt', current_path + 'unique_sid.txt')
        shutil.copyfile(data_dir + 'unique_uid.txt', current_path + 'unique_uid.txt')

        # Copy the features in each subfolder
        shutil.copyfile(data_dir + 'features.csv', current_path + 'features.csv')

    return


def numerize_warm(data_dir='data/', n_splits=10):

    # Load the user and song to id mappings
    with open(data_dir + 'user2id.json', 'r') as f:
        user2id = json.load(f)
    with open(data_dir + 'song2id.json', 'r') as f:
        song2id = json.load(f)

    # Numerize all the subsets / splits
    setting = 'warm'
    for i_split in range(n_splits):
        for subset_to_numerize in ['train', 'test', 'val']:
            current_path = data_dir + setting + '/split' + str(i_split) + '/' + subset_to_numerize

            # TP data
            data_tp = pd.read_csv(current_path + '_tp.csv')
            uid = list(map(lambda x: user2id[x], data_tp['uid']))
            sid = list(map(lambda x: song2id[x], data_tp['sid']))
            data_tp['uid'] = uid
            data_tp['sid'] = sid
            data_tp.to_csv(current_path + '_tp.num.csv', index=False)

        # Features
        current_path = data_dir + setting + '/split' + str(i_split) + '/'
        data_feats = pd.read_csv(current_path + 'features.csv')
        sid = list(map(lambda x: song2id[x], data_feats['sid']))
        data_feats = data_feats.assign(sid=sid)
        data_feats.to_csv(current_path + 'feats.num.csv', index=False)

    return


if __name__ == '__main__':

    MIN_USER_COUNT, MIN_SONG_COUNT, MIN_COUNT = 20, 50, 7
    data_dir = 'data/'
    n_splits = 10

    # Load the TP data and filter out inactive data
    #load_filter_record_tp(data_dir, min_uc=MIN_USER_COUNT, min_sc=MIN_SONG_COUNT, min_c=MIN_COUNT)

    # Process the song features
    #load_record_features(data_dir)

    # Update the TP songs (to keep only those for which there are available features), and print the density level
    #update_tp_record(data_dir)
    #density_level = print_density_level(data_dir)

    # Create the various splits (and numerize files) for the cold-start scenario
    split_cold(data_dir=data_dir, n_splits=n_splits)
    numerize_cold(data_dir=data_dir, n_splits=n_splits)

    # Same for the warm-start scenario
    #split_warm(data_dir=data_dir, n_splits=n_splits)
    #numerize_warm(data_dir=data_dir, n_splits=n_splits)

# EOF
