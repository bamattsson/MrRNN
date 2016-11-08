import numpy as np
from tensorflow.contrib import learn

def generate_data_ubuntu(data_set = 'training', max_utterances = 6, max_tokens = 20, min_frequency = 10):
    """Generates data set from ubuntu conversation history.

    Requieres unzipped version of www.iulianserban.com/Files/UbuntuDialogueCorpus.zip placed in map ./data

    NB: each __eot__ token is considered to end the turn (and what in the article is called utterance). __eou__ is just a pause and therefore considered to be a token without any specific meaning.

    Returns:
    x_data -- 3-dim matrix with [num_data, max_utterances, max_token], padded in the beggining of each utterance
    y_data -- as x_data but padded in the end of each utterance
    vocab_processor -- vocab_processor containing token-value encoding
    """
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
    y_data = np.array(data) # Paddings in the end

    # Create x_data that has paddings in the beginning
    x_data = np.zeros_like(y_data)
    for i in range(y_data.shape[0]):
        for j in range(y_data.shape[1]):
            for k in range(y_data.shape[2]-1,-1,-1):
                if y_data[i,j,k] != 0:
                    x_data[i,j,-(k+1):] = y_data[i,j,:k+1]
                    break

    return x_data, y_data, vocab_processor

def generate_test_data_sequence(examples=50000, num_seq = 4, num_steps = 20):
    """Generate a simple test sequence.

    From the example at: http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
    """
    X = np.zeros([examples, num_seq, num_steps])
    for i in range(examples):
        for j in range(num_seq*num_steps):
            jj, kk = divmod(j,num_steps)
            if ((j + 1) % num_steps == 0):
                X[i,jj, kk] = 2 # EOL character
                continue
            threshold = 0.5
            j3, k3 = divmod(j - 3, num_steps)
            j8, k8 = divmod(j - 8, num_steps)
            if j >= 3 and X[i,j3, k3] == 1:
                threshold += 0.5
            if j >= 8 and X[i,j8, k8] == 1:
                threshold -= 0.25
            if np.random.rand() > threshold:
                X[i,jj,kk] = 0
            else:
                X[i,jj,kk] = 1
    return X

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Generates a batch iterator for a dataset."""
    data = np.array(data)
    data_size = len(data)
    if len(data) % batch_size == 0:
        num_batches_per_epoch = int(len(data) / batch_size)
    else:
        num_batches_per_epoch = int(len(data) / batch_size) + 1
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
            yield data[indices[start_index:end_index]]

def random_embedding(vocabulary_size, embedding_dim = 128, value_range = .1):
    """Creates word embedding matrix from scratch."""                                           
    W_embeddings = np.random.uniform(-value_range,value_range, (vocabulary_size, embedding_dim))
    return W_embeddings

def _flatten_list(list_):
    for sublist in list_:
        for str_ in sublist:
            yield str_
