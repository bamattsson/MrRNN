import numpy as np
import tensorflow as tf

def build_graph(coarse_kwargs, n_hidden_coarse_prediction, nl_kwargs, learning_rate, gradient_clipping=0):
    """Constructs the graph.

    Arguments:
    coarse_kwargs -- arguments to the coarse HRED graph. See _build_HRED_graph for what to include.
    learning_rate -- learning rate to use in Adam optimizer

    Keyword arguments:
    gradient_clipping -- (default = 0) if active (i.e. > 0) gradients will be confined to (-gradient_clipping, gradient_clipping) in each dimension
    """
    # Input variables
    # TODO: variable batch_size, i.e. use None
    coarse_sequence_input = tf.placeholder(tf.int32, [coarse_kwargs['batch_size'], coarse_kwargs['num_seq'], coarse_kwargs['num_steps']])
    coarse_length_input = tf.placeholder(tf.int32, [coarse_kwargs['batch_size'], coarse_kwargs['num_seq']]) # how many num_steps each utterance contains
    coarse_vocab_input = tf.placeholder(tf.float32, shape = coarse_kwargs['embedding_shape'])
    coarse_W_embedding = tf.Variable(coarse_vocab_input, name='W') # [vocab_size, emb_dim]

    nl_sequence_input = tf.placeholder(tf.int32, [nl_kwargs['batch_size'], nl_kwargs['num_seq'], nl_kwargs['num_steps']])
    nl_length_input = tf.placeholder(tf.int32, [nl_kwargs['batch_size'], nl_kwargs['num_seq']]) # how many num_steps each utterance contains
    nl_vocab_input = tf.placeholder(tf.float32, shape = nl_kwargs['embedding_shape'])
    nl_W_embedding = tf.Variable(nl_vocab_input, name='W') # [vocab_size, emb_dim]

    # Build coarse HRED graph
    with tf.variable_scope('coarse_sub-model'), tf.name_scope('coarse_sub-model'):
        coarse_losses, _ = _build_HRED_graph(coarse_sequence_input, coarse_length_input, coarse_W_embedding, **coarse_kwargs)

    # Build coarse prediction encoder graph
    with tf.variable_scope('coarse_pred_encoder'), tf.name_scope('coarse_pred_encoder'):
        sliced_coarse_seq_input = tf.slice(coarse_sequence_input,[0,1,0],[-1,-1,-1])
        sliced_coarse_len_input = tf.slice(coarse_length_input,[0,1],[-1,-1])
        coarse_prediction_states = _build_coarse_prediction_encoder(sliced_coarse_seq_input, sliced_coarse_len_input, coarse_W_embedding, n_hidden_coarse_prediction)

    # Build natural language HRED graph
    with tf.variable_scope('nl_sub-model'), tf.name_scope('nl_sub-model'):
        nl_losses, _ = _build_HRED_graph(nl_sequence_input, nl_length_input, nl_W_embedding, **nl_kwargs, coarse_prediction_states = coarse_prediction_states)

    # Do training
    losses = tf.concat(0, [coarse_losses, nl_losses])
    total_loss = tf.reduce_mean(losses)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gvs = optimizer.compute_gradients(total_loss)
    if gradient_clipping > 0:
        gvs = [(tf.clip_by_value(grad, -1.*gradient_clipping, 1.*gradient_clipping), var) for grad, var in gvs]
    train_step = optimizer.apply_gradients(gvs)

    # Creating dict that grants access to important nodes in the graph
    graph_nodes = {'coarse_vocab_input' : coarse_vocab_input,
        'coarse_sequence_input' : coarse_sequence_input,
        'coarse_length_input' : coarse_length_input,
        'nl_vocab_input' : nl_vocab_input,
        'nl_sequence_input' : nl_sequence_input,
        'nl_length_input' : nl_length_input,
        'total_loss' : total_loss,
        'train_step' : train_step
        }
    return graph_nodes

def _build_coarse_prediction_encoder(sequence_input, length_input, W_embedding, n_hidden):
    """Constructs the coarse prediction encoder.

    Arguments:
    sequence_input -- coarse sequence only including the relevant sequences [batch_size, num_seq-1, num_steps]
    length_input -- num steps of each sequence [batch_size, num_seq-1]
    W_embedding -- embedding matrix variable
    """
    # Pre-process data
    sequence_data = tf.nn.embedding_lookup(W_embedding, sequence_input) # [batch_size, num_seq-1, num_steps, emb_dim]
    sequence_data = tf.unpack(sequence_data, axis=1) # list(num_seq-1*[batch_size, num_steps, emb_dim])
    length_data = tf.unpack(length_input, axis=1) # list(num_seq-1*[batch_size])

    # Do RNN
    cell = tf.nn.rnn_cell.GRUCell(n_hidden)
    init_state = tf.zeros([int(length_data[0].get_shape()[0]), n_hidden])
    final_states = []
    for (z_seq, z_len) in zip(sequence_data, length_data):
        z = tf.unpack(z_seq, axis=1)
        _, init_state = tf.nn.rnn(cell, z, initial_state = init_state, sequence_length = z_len)
        tf.get_variable_scope().reuse_variables()
        final_states.append(init_state)
    final_states = tf.pack(final_states, axis=1) # [batch_size, num_seq-1]
    return final_states

def _build_HRED_graph(sequence_input, length_input, W_embedding, embedding_shape, num_seq, num_steps, batch_size, n_hidden_encoder, n_hidden_context, n_hidden_decoder, coarse_prediction_states = None):
    """Constructs a HRED graph.

    Arguments:
    sequence_input -- sequence data [batch_size, num_seq, num_steps]
    length_input -- num steps of each sequence [batch_size, num_seq]
    W_embedding -- embedding matrix variable
    embedding_shape -- shape of word embedding [voc_size, num_dim]
    num_seq -- max number of sequences
    num_steps -- max number of steps in each sequence
    batch_size -- number of batches
    n_hidden_encoder -- hiden state size encoder rnn
    n_hidden_context -- hiden state size context rnn
    n_hidden_decoder -- hiden state size decoder rnn

    Keyword arguments:
    coarse_prediction_states -- (default None) if desired should be a tensor [batch_size, num_seq-1, coarse_emb_dim]
    """
    # TODO:
    # a lot of unpacking/packing of variables, probably not very efficient
    # choose init values for initialization

    # Get embedding
    with tf.device('/cpu:0'), tf.name_scope('embedding'):
        # TODO: paper wants different i/o embeddings?
        # TODO: this could be done outside of this function to save some time (both this and coarse_pred_encoder does this)
        emb_dim = embedding_shape[1]
        sequence_data = tf.nn.embedding_lookup(W_embedding, sequence_input) # [batch_size, num_seq, num_steps, emb_dim]
        sequence_data = tf.unpack(sequence_data, axis=1) # list(num_seq*[batch_size, num_steps, emb_dim])

    with tf.name_scope('transform_data'):
        length_data = tf.unpack(length_input, axis=1) # list(num_seq*[batch_size])

        # Process x
        x_data = sequence_data[:-1] # list(num_seq-1*[batch_size, num_steps, emb_dim])
        x_data_length = length_data[:-1] # list(num_seq-1*[batch_size])

        # Process y
        y_data = sequence_data[1:] # list(num_seq-1*[batch_size, num_steps, emb_dim])
        y_data_length = length_data[1:] # list(num_seq-1*[batch_size])
        y_data = [tf.unpack(y_seq, axis = 1) for y_seq in y_data]
        y_data = [[tf.zeros([batch_size, emb_dim])] + y_seq[:-1] for y_seq in y_data]
        # y_data: list(num_seq - 1 *list(num_steps * [batch_size, emd_dim]))

    # Encoder RNN
    with tf.name_scope('encoder'):
        final_states_enc = _build_encoders(x_data, x_data_length, n_hidden_encoder, batch_size)

    # Context RNN
    with tf.name_scope('context') as con_scope:
        context_length = tf.cast(tf.greater(length_input, 0), dtype=tf.int32)
        context_length = tf.reduce_sum(context_length, 1)
        init_state_con = tf.zeros([batch_size, n_hidden_context])
        context_cell = tf.nn.rnn_cell.GRUCell(n_hidden_context)
        output_context, _ = tf.nn.rnn(context_cell, final_states_enc, initial_state = init_state_con, sequence_length = context_length)
        # output_context: list(num_seq - 1 *[batch_size, n_hidden_con])

    # Context to decoder
    if coarse_prediction_states is None:
        W_con2dec = tf.get_variable('W_con2dec', [n_hidden_context, n_hidden_decoder])
        b_con2dec = tf.get_variable('b_con2dec', [n_hidden_decoder])
        init_state_dec = [tf.tanh(tf.matmul(fs, W_con2dec) + b_con2dec) for fs in output_context]
    else:
        coarse_prediction_states = tf.unpack(coarse_prediction_states,axis=1) # list(num_seq-1 *[batch_size, coarse_emb_dim]
        word_coarse_concatenated = [tf.concat(1, [w_e, c_p_e]) for w_e, c_p_e in zip(output_context, coarse_prediction_states)]

        W_con2dec = tf.get_variable('W_con2dec', [tf.Dimension(n_hidden_context) + coarse_prediction_states[0].get_shape()[1], n_hidden_decoder])
        b_con2dec = tf.get_variable('b_con2dec', [n_hidden_decoder])
        init_state_dec = [tf.tanh(tf.matmul(fs, W_con2dec) + b_con2dec) for fs in word_coarse_concatenated]

    # Decoder RNN
    with tf.name_scope('decoder'):
        output_dec = _build_decoders(y_data, y_data_length, n_hidden_decoder, init_state_dec, batch_size, emb_dim)
    # output_dec: list(num_seq - 1 *list(num_steps *[batch_size, n_hidden_dec])

    # To output
    with tf.name_scope('output'):
        W_out = tf.get_variable('W_out', [n_hidden_decoder, emb_dim])
        E_out = tf.get_variable('E_out', [emb_dim, emb_dim])
        b_out = tf.get_variable('b_out', [emb_dim])

        output_dec = sum(output_dec,[]) # list((num_seq - 1)*(num_steps)*[batch_size, n_hidden_dec])
        y_data = sum(y_data,[])
        logits = [tf.matmul(o, W_out) + tf.matmul(y, E_out) + b_out for o, y in zip(output_dec, y_data)]
        logits_words = [tf.matmul(l, W_embedding, transpose_b = True) for l in logits] #TODO: not very efficient, implement(?): http://sebastianruder.com/word-embeddings-softmax/
        logits_words = tf.pack(logits_words,axis=1)
        logits_words = tf.reshape(logits_words, [-1, embedding_shape[0]])

    # Calculate loss
    with tf.name_scope('loss'):
        y_data_input = tf.slice(sequence_input, [0,1,0], [-1,-1,-1])
        y_data_input_flat = tf.reshape(y_data_input, [-1])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_words, y_data_input_flat)
        not_pad_unk_mask = tf.not_equal(y_data_input_flat, tf.constant(0, tf.int32))
        losses = tf.boolean_mask(losses, not_pad_unk_mask) # Not count loss from unks and padding
        total_loss = tf.reduce_mean(losses)

    return losses, total_loss

def _build_encoders(x_sequences, x_length, n_hidden_enc, batch_size):
    # TODO: bidirectional?
    # x_sequences - list(num_seq - 1 *[batch_size, num_steps, emb_dim])
    with tf.variable_scope('encoder'):
        init_state = tf.zeros([batch_size, n_hidden_enc])
        cell = tf.nn.rnn_cell.GRUCell(n_hidden_enc)
        final_states = []
        for (x_seq, x_len) in zip(x_sequences, x_length):
            x = tf.unpack(x_seq, axis = 1) # list(num_steps * [batch_size, emb_dim])
            _, final_state = tf.nn.rnn(cell, x, initial_state = init_state, sequence_length = x_len)
            tf.get_variable_scope().reuse_variables()
            final_states.append(final_state)
    return final_states

def _build_decoders(y_sequences, y_length, n_hidden_dec, init_states, batch_size, emb_dim):
    # TODO: bidirectional?
    # y_sequences - list(num_seq - 1 list(num_steps*[batch_size, num_steps, emb_dim])
    with tf.variable_scope('decoder'):
        cell = tf.nn.rnn_cell.GRUCell(n_hidden_dec)
        outputs_list = []
        for (y_seq, y_len, init_s) in zip(y_sequences, y_length, init_states):
            outputs, _ = tf.nn.rnn(cell, y_seq, initial_state = init_s, sequence_length = y_len)
            tf.get_variable_scope().reuse_variables()
            outputs_list.append(outputs)
    return outputs_list
