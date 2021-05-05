
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
