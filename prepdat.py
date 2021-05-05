#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import pandas as pd
import arff
from sklearn.preprocessing import scale
import shutil


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'count']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def print_density_level(data_dir='data/'):

    tp = pd.read_csv(data_dir + 'tp.csv')
    usercount, songcount = get_count(tp, 'uid'), get_count(tp, 'sid')
    density_level = float(tp.shape[0]) / (usercount.shape[0] * songcount.shape[0])
    print("After filtering, there are %d triplets from %d users and %d songs (sparsity level %.3f%%)"
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

    # Only keep the triplets for songs which were listened to by at least min_sc users, and at least min_sc times
    songcount = get_count(tp, 'sid')
    tp = tp[tp['sid'].isin(songcount.index[songcount.values >= min_sc])]
    usercount = get_count(tp, 'uid')
    tp = tp[tp['uid'].isin(usercount.index[usercount.values >= min_uc])]

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
        print(ii)
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


def update_tp_record(data_dir='data/'):

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


def split_tp_out(data_dir='data/'):

    # Taste profile data
    tp = pd.read_csv(data_dir + 'tp.csv')

    # List of unique songs
    unique_sid = []
    with open(data_dir + '/unique_sid.txt', 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    # Make a 70/20/10 split for train/val/test
    n_songs = len(unique_sid)
    train_sid = unique_sid[:int(0.7 * n_songs)]
    val_sid = unique_sid[int(0.7 * n_songs):int(0.9 * n_songs)]
    test_sid = unique_sid[int(0.9 * n_songs):]

    # Generate in and out of matrix split from TP
    train_tp = tp[tp['sid'].isin(train_sid)]
    val_tp = tp[tp['sid'].isin(val_sid)]
    test_tp = tp[tp['sid'].isin(test_sid)]

    # Save the .csv
    train_tp.to_csv(data_dir + 'out/train_tp.csv', index=False)
    val_tp.to_csv(data_dir + 'out/val_tp.csv', index=False)
    test_tp.to_csv(data_dir + 'out/test_tp.csv', index=False)

    return


def split_tp_in(data_dir='data/'):

    # Taste profile data
    tp = pd.read_csv(data_dir + 'tp.csv')

    # List of unique songs
    unique_sid = []
    with open(data_dir + '/unique_sid.txt', 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    # Pick out 10% of the rating for in-matrix testing
    n_ratings = tp.shape[0]
    test = np.random.choice(n_ratings, size=int(0.1 * n_ratings), replace=False)
    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[test] = True
    test_tp = tp[test_idx]
    tp_notest = tp[~test_idx]

    # Pick out 20% of the (remaining) ratings as validation set
    n_ratings = tp_notest.shape[0]
    val = np.random.choice(n_ratings, size=int(0.2/0.9 * n_ratings), replace=False)
    val_idx = np.zeros(n_ratings, dtype=bool)
    val_idx[val] = True
    val_tp = tp_notest[val_idx]
    train_tp = tp_notest[~val_idx]

    # Save the .csv
    train_tp.to_csv(data_dir + 'in/train_tp.csv', index=False)
    val_tp.to_csv(data_dir + 'in/val_tp.csv', index=False)
    test_tp.to_csv(data_dir + 'in/test_tp.csv', index=False)

    return


def numerize_tp(data_dir='data/'):

    # Load the user and song to id mappings
    with open(data_dir + 'user2id.json', 'r') as f:
        user2id = json.load(f)
    with open(data_dir + 'song2id.json', 'r') as f:
        song2id = json.load(f)

    # Numerize all the TP subsets
    for in_out in ['out/', 'in/']:
        for subset_to_numerize in ['train_tp', 'test_tp', 'val_tp']:
            data_tp = pd.read_csv(data_dir + in_out + subset_to_numerize + '.csv')
            uid = list(map(lambda x: user2id[x], data_tp['uid']))
            sid = list(map(lambda x: song2id[x], data_tp['sid']))
            data_tp['uid'] = uid
            data_tp['sid'] = sid
            data_tp.to_csv(data_dir + in_out + subset_to_numerize + '.num.csv', index=False)

    return


def split_numerize_features_out(data_dir='data/'):

    # Load features
    my_features = pd.read_csv(data_dir + 'features.csv')

    # List of unique songs
    unique_sid = []
    with open(data_dir + '/unique_sid.txt', 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    # Dic for numerization
    with open(data_dir + 'song2id.json', 'r') as f:
        song2id = json.load(f)

    # Make a 70/20/10 split for train/val/test
    n_songs = len(unique_sid)
    train_sid = unique_sid[:int(0.7 * n_songs)]
    val_sid = unique_sid[int(0.7 * n_songs):int(0.9 * n_songs)]
    test_sid = unique_sid[int(0.9 * n_songs):]

    # Generate in and out of matrix split from features
    train_feats = my_features[my_features['sid'].isin(train_sid)]
    val_feats = my_features[my_features['sid'].isin(val_sid)]
    test_feats = my_features[my_features['sid'].isin(test_sid)]

    train_feats.to_csv(data_dir + 'out/train_feats.csv', index=False)
    val_feats.to_csv(data_dir + 'out/val_feats.csv', index=False)
    test_feats.to_csv(data_dir + 'out/test_feats.csv', index=False)

    # Numerize and record
    sid_train = list(map(lambda x: song2id[x], train_feats['sid']))
    train_feats = train_feats.assign(sid=sid_train)
    train_feats.to_csv(data_dir + 'out/train_feats.num.csv', index=False)
    sid_val = list(map(lambda x: song2id[x], val_feats['sid']))
    val_feats = val_feats.assign(sid=sid_val)
    val_feats.to_csv(data_dir + 'out/val_feats.num.csv', index=False)
    sid_test = list(map(lambda x: song2id[x], test_feats['sid']))
    test_feats = test_feats.assign(sid=sid_test)
    test_feats.to_csv(data_dir + 'out/test_feats.num.csv', index=False)

    return


def numerize_features_in(data_dir='data/'):

    # Load features
    my_features = pd.read_csv(data_dir + 'features.csv')

    # Dic for numerization
    with open(data_dir + 'song2id.json', 'r') as f:
        song2id = json.load(f)

    # Numerize and record
    sid_features = list(map(lambda x: song2id[x], my_features['sid']))
    my_features = my_features.assign(sid=sid_features)
    my_features.to_csv(data_dir + 'in/feats.num.csv', index=False)

    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12345)

    MIN_USER_COUNT, MIN_SONG_COUNT, MIN_COUNT = 20, 50, 7
    data_dir = 'data/'

    # Load the TP data and filter out inactive data
    load_filter_record_tp(data_dir, min_uc=MIN_USER_COUNT, min_sc=MIN_SONG_COUNT, min_c=MIN_COUNT)

    # Process the song features
    load_record_features(data_dir)

    # Update the TP songs (to keep only those for which there are available features)
    update_tp_record(data_dir)

    # Print the density level
    print_density_level(data_dir)

    # Create train / validation / test split for in and out recommendation and numerize the song and user ids
    split_tp_out(data_dir)
    split_tp_in(data_dir)
    numerize_tp(data_dir)

    # Same for the features (no need for splitting them for 'in' recommendation)
    split_numerize_features_out(data_dir)
    numerize_features_in(data_dir)

    # Copy the list of user/song IDs in both directory
    shutil.copyfile(data_dir + 'unique_sid.txt', data_dir + 'out/unique_sid.txt')
    shutil.copyfile(data_dir + 'unique_sid.txt', data_dir + 'in/unique_sid.txt')
    shutil.copyfile(data_dir + 'unique_uid.txt', data_dir + 'out/unique_uid.txt')
    shutil.copyfile(data_dir + 'unique_uid.txt', data_dir + 'in/unique_uid.txt')

# EOF
