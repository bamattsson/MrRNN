import numpy as np

def gen_data_sequence(examples=50000, num_seq = 4, num_steps = 20):
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

def random_embedding(vocabulary_size, embedding_dim = 128, value_range = 1.):
    """Creates word embedding matrix from scratch."""                                           
    W_embeddings = np.random.uniform(-value_range,value_range, (vocabulary_size, embedding_dim))
    return W_embeddings
