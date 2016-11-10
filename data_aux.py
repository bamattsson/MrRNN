import gc
import numpy as np
from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec

def generate_data_ubuntu(data_set = 'training', max_utterances = 6, max_tokens = 20, vocab_processor = None, min_frequency = 10):
    """Generates data set from ubuntu conversation history.

    Requieres unzipped version of www.iulianserban.com/Files/UbuntuDialogueCorpus.zip placed in map ./data

    NB: each __eot__ token is considered to end the turn (and what in the article is called utterance). __eou__ is just a pause and therefore considered to be a token without any specific meaning.

    Returns:
    data -- 3-dim matrix with [num_data, max_utterances, max_token], padded in the end of each utterance
    data_len -- 2-dim matrix [num_data, max_utterances], length of each utterance
    vocab_processor -- vocab_processor containing token-value encoding
    """
    # TODO: add argument vocab_processor to be able to generate test with same embedding
    # Get data from file
    data_str = []
    path = './data/raw_' + data_set + '_text.txt'
    with open(path) as f:
        for line in f:
            utterances = line.split('__eot__ ')
            len_conversation = len(utterances)
            if len_conversation > max_utterances:
                continue
            row = []
            too_long = False
            for u in utterances:
                if len(u.split()) > max_tokens:
                    too_long = True
                    break
                tmp_u = u[:-9] + " __eot__" # Changing last __eou__ to __eot__
                row.append(tmp_u)
            if not too_long:
                data_str.append(row)

    # Create token-value encoding
    if vocab_processor is None:
        # TODO: change min_frequency to max_vocab_size in alignment with article
        list_iterator = _flatten_list(data_str)
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_tokens, min_frequency = min_frequency)
        vocab_processor = vocab_processor.fit(list_iterator)

    # Transform data tokens-values
    data = []
    for row in data_str:
        tmp = np.array(list(vocab_processor.transform(row)))
        tmp = np.pad(tmp,[[0, max_utterances - len(tmp)],[0,0]], 'constant')
        data.append(tmp)
    data = np.array(data) # Paddings in the end

    # Create data_len that contains information about how long each utterance is
    data_len = np.zeros(data.shape[:2])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]-1,-1,-1):
                if data[i,j,k] != 0:
                    data_len[i,j] = k + 1
                    break

    return data, data_len, vocab_processor

def generate_test_data_sequence(examples=50000, num_seq = 4, num_steps = 20):
    """Generate a simple test sequence.

    Data sequence with either 1 or 2 (3 is end of turn)

    From the example at: http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
    """
    X = np.zeros([examples, num_seq, num_steps])
    for i in range(examples):
        for j in range(num_seq*num_steps):
            jj, kk = divmod(j,num_steps)
            if ((j + 1) % num_steps == 0):
                X[i,jj, kk] = 3 # EOL character
                continue
            threshold = 0.5
            j3, k3 = divmod(j - 3, num_steps)
            j8, k8 = divmod(j - 8, num_steps)
            if j >= 3 and X[i,j3, k3] == 2:
                threshold += 0.5
            if j >= 8 and X[i,j8, k8] == 2:
                threshold -= 0.25
            if np.random.rand() > threshold:
                X[i,jj,kk] = 1
            else:
                X[i,jj,kk] = 2

    L = num_steps*np.ones([examples, num_seq])
    return X, L

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Generates a batch iterator for a dataset.

    Arguments:
    data -- tuple of np.ndarrays containing the data
    """
    data_size = len(data[0])
    if len(data[0]) % batch_size == 0:
        num_batches_per_epoch = int(len(data[0]) / batch_size)
    else:
        num_batches_per_epoch = int(len(data[0]) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            indices = np.random.permutation(np.arange(data_size))
        else:
            indices = np.arange(data_size)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if end_index - start_index != batch_size:
                continue
            yield tuple([d[indices[start_index:end_index]] for d in data])

def pretrained_embedding(vocab_processor):
    """Creates word embedding matrix from GoogleNews w2v.

    Requieres google news w2v downloaded from https://code.google.com/archive/p/word2vec/ in data
    """
    w2v = Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
    w2v.init_sims(replace=True)
    gc.collect()
    words = [vocab_processor.vocabulary_.reverse(i) for i in range(vocab_processor.vocabulary_.__len__())]

    W_embeddings = []
    for w in words:
        try:
            W_embeddings.append(w2v.__getitem__(w))
        except KeyError:
            W_embeddings.append(np.random.uniform(-0.1, 0.1, 300)) # Boundries makes variance equal as the ones from google
    del w2v
    gc.collect()
    W_embeddings = np.array(W_embeddings)
    return W_embeddings

def random_embedding(vocab_processor, embedding_dim = 128, value_range = .1):
    """Creates word embedding matrix from scratch."""
    W_embeddings = np.random.uniform(-value_range,value_range, (len(vocab_processor.vocabulary_), embedding_dim))
    return W_embeddings

def _flatten_list(list_):
    for sublist in list_:
        for str_ in sublist:
            yield str_
