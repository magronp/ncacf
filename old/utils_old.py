
def init_weights(m):
    if type(m) == Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def init_ncacf(my_model, path_pretrain, variant='relaxed'):

    # Load the pretrained deep content model and WMF
    model_pretrain = torch.load(os.path.join(path_pretrain, 'model.pt'))

    # Initialize the deep content extractor
    my_model.fnn_in = model_pretrain.fnn_in
    my_model.fnn_hi1 = model_pretrain.fnn_hi1
    my_model.fnn_out = model_pretrain.fnn_out

    # Initialize the user and item embedding (if any)
    my_model.user_emb = model_pretrain.user_emb
    if variant == 'relaxed':
        my_model.item_emb = model_pretrain.item_emb

    return my_model


def init_model_joint(my_model, path_pretrain, variant='relaxed'):

    # Load the pretrained deep content model and WMF
    model_pretrain = torch.load(os.path.join(path_pretrain, 'model.pt'))
    wmf = np.load(os.path.join(path_pretrain, 'wmf.npz'))

    # Initialize the deep content extractor
    my_model.fnn_in = model_pretrain.fnn_in
    my_model.fnn_hi1 = model_pretrain.fnn_hi1
    my_model.fnn_out = model_pretrain.fnn_out

    # Initialize the user and item embedding (if any)
    my_model.user_emb = torch.nn.Embedding.from_pretrained(torch.tensor(wmf['W']))
    if variant == 'relaxed':
        my_model.item_emb = torch.nn.Embedding.from_pretrained(torch.tensor(wmf['H']))

    return my_model


def gen_neg_tp(data_dir='data/', in_out='out', neg_ratio=5):

    path_data = data_dir + in_out + '/'

    n_users = len(open(path_data + 'unique_uid.txt').readlines())
    n_songs_train = len(open(path_data + 'unique_sid.txt').readlines())
    if in_out == 'out':
        n_songs_train = int(0.7 * n_songs_train)
    list_items_total = np.arange(n_songs_train)

    # TP data
    train_tp_data = pd.read_csv(path_data + 'train_tp.num.csv')
    val_tp_data = pd.read_csv(path_data + 'val_tp.num.csv')
    test_tp_data = pd.read_csv(path_data + 'test_tp.num.csv')
    list_users_train = pd.unique(train_tp_data['uid'])

    # Neg sampling item lists
    n_songs_neg = neg_ratio - 1
    item_neg_sampling = np.zeros((n_users, n_songs_neg))

    # Negative sampling of items for each user
    for u in list_users_train:
        list_items_u_train = np.unique(train_tp_data[train_tp_data['uid'] == u]['sid'])
        list_items_samples_u = np.delete(list_items_total, list_items_u_train)
        if in_out == 'in':
            list_items_u_val = np.unique(val_tp_data[val_tp_data['uid'] == u]['sid'])
            list_items_u_test = np.unique(test_tp_data[test_tp_data['uid'] == u]['sid'])
            list_items_samples_u = np.delete(list_items_samples_u, list_items_u_val)
            list_items_samples_u = np.delete(list_items_samples_u, list_items_u_test)
        item_neg_sampling[u, :] = np.random.choice(list_items_samples_u, n_songs_neg, replace=False)

    # Store the result
    np.savez(path_data + 'train_tp_neg.num.npz', neg_items=item_neg_sampling)

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



def split_tp_in(data_dir='data/', n_splits=10):

    # Load the Taste profile data
    tp = pd.read_csv(data_dir + 'tp.csv')

    # List of unique songs
    unique_sid = []
    with open(data_dir + '/unique_sid.txt', 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    # Pick out 10% of the binarized playcounts for in-matrix testing
    n_ratings = tp.shape[0]
    test = np.random.choice(n_ratings, size=int(0.1 * n_ratings), replace=False)
    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[test] = True
    test_tp = tp[test_idx]
    tp_notest = tp[~test_idx]

    # Pick out 20% of the (remaining) binarized playcounts  as validation set
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


class DatasetAttributesNegsamp(Dataset):

    def __init__(self, features_path, tp_path, tp_neg, n_users, n_songs):

        # Acoustic content features
        features = pd.read_csv(features_path).to_numpy()
        features = features[features[:, 0].argsort()]
        x = np.delete(features, 0, axis=1)
        self.x = torch.tensor(x).float()

        # TP data
        tp_data = load_tp_data_sparse(tp_path, n_users, n_songs)
        tp_data = tp_data.coalesce()
        self.us, self.it = tp_data.indices()
        self.count = tp_data.values()

        # Also load the list of negative samples (=items) per user
        self.neg_items = torch.tensor(np.load(tp_neg)['neg_items'], dtype=torch.long)

    def __len__(self):
        return self.count.__len__()

    def __getnegitem__(self, us_pos):
        return self.neg_items[us_pos, :]

    def __getitem__(self, data_point):

        us_pos, it_pos, count_pos = self.us[data_point], self.it[data_point], self.count[data_point]
        it_neg = self.__getnegitem__(us_pos)

        return self.x[it_pos, :], self.x[it_neg, :], us_pos, count_pos, it_pos, it_neg

def load_tp_data_sparse(csv_file, n_users, n_songs):

    tp = pd.read_csv(csv_file)
    indices = torch.tensor([tp['uid'], tp['sid']], dtype=torch.long)
    #values = torch.ones_like(torch.tensor(tp['count'], dtype=torch.float32))
    values = torch.tensor(tp['count'], dtype=torch.float32)

    # Construct a sparse tensor
    tp_sparse_tens = torch.sparse.FloatTensor(indices, values, torch.Size([n_users, n_songs]))

    return tp_sparse_tens


def load_tp_data_old(csv_file, shape, alpha=2.0, epsilon=1e-6):
    tp = pd.read_csv(csv_file)
    rows, cols = np.array(tp['uid'], dtype=np.int32), np.array(tp['sid'], dtype=np.int32)
    count = tp['count']
    sparse_tp = scipy.sparse.csr_matrix((count, (rows, cols)), dtype=np.int16, shape=shape)
    # Binarize the playcounts
    sparse_tp.data = np.ones_like(sparse_tp.data)
    # Get the confidence from binarized playcounts directly
    conf = sparse_tp.copy()
    conf.data[conf.data == 0] = 0.01
    conf.data -= 1  # conf surplus (used in WMF implementation)
    return sparse_tp, rows, cols, conf


def log_surplus_confidence_matrix(R, alpha, epsilon):
    # To construct the surplus confidence matrix, we need to operate only on
    # the nonzero elements.
    # This is not possible: C = alpha * np.log(1 + R / epsilon)
    C = R.copy()
    C.data = alpha * np.log(1 + C.data / epsilon)
    return C


def numerize_features_warm(data_dir='data/'):

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