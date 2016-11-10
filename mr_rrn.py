import numpy as np
import tensorflow as tf

def build_graph(hred_kwargs, learning_rate, gradient_clipping=0):
    """Constructs the graph.

    Arguments:
    hred_kwargs -- arguments to the HRED graph. See _build_HRED_graph for what to include.
    learning_rate -- learning rate to use in Adam optimizer

    Keyword arguments:
    gradient_clipping -- (default = 0) if active (i.e. > 0) gradients will be confined to (-gradient_clipping, gradient_clipping) in each dimension
    """
    # Build HRED graph
    (vocab_input, sequence_input, length_input, total_loss) = _build_HRED_graph(**hred_kwargs)

    # Do training
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gvs = optimizer.compute_gradients(total_loss)
    if gradient_clipping > 0:
        gvs = [(tf.clip_by_value(grad, -1.*gradient_clipping, 1.*gradient_clipping), var) for grad, var in gvs]
    train_step = optimizer.apply_gradients(gvs)

    # Creating dict that grants access to important nodes in the graph, this to avoid confusion with names
    graph_nodes = {'vocab_input' : vocab_input,
        'sequence_input' : sequence_input,
        'length_input' : length_input,
        'total_loss' : total_loss,
        'train_step' : train_step
        }
    return graph_nodes

def _build_HRED_graph(embedding_shape, num_seq, num_steps, batch_size, n_hidden_encoder, n_hidden_context, n_hidden_decoder):
    """Constructs a HRED graph.

    Arguments:
    embedding_shape -- shape of word embedding [voc_size, num_dim]
    num_seq -- max number of sequences
    num_steps -- max number of steps in each sequence
    batch_size -- number of batches
    n_hidden_encoder -- hiden state size encoder rnn
    n_hidden_context -- hiden state size context rnn
    n_hidden_decoder -- hiden state size decoder rnn
    """
    # TODO:
    # a lot of unpacking/packing of variables, probably not very efficient
    # choose init values for initialization

    # Input variables
    # TODO: variable batch_size
    sequence_input = tf.placeholder(tf.int32, [batch_size, num_seq, num_steps])
    length_input = tf.placeholder(tf.int32, [batch_size, num_seq]) # how many num_steps each utterance contains

    # Get embedding
    with tf.device('/cpu:0'), tf.name_scope('embedding'):
        # TODO: paper wants different i/o embeddings?
        vocab_input = tf.placeholder(tf.float32, shape = embedding_shape)
        emb_dim = embedding_shape[1]
        W_embedding = tf.Variable(vocab_input, name='W') # [vocab_size, emb_dim]
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
    final_states_enc = _build_encoders(x_data, x_data_length, n_hidden_encoder, batch_size)

    # Context RNN
    with tf.variable_scope('context') as con_scope:
        context_length = tf.cast(tf.greater(length_input, 0), dtype=tf.int32)
        context_length = tf.reduce_sum(context_length, 1)
        init_state_con = tf.zeros([batch_size, n_hidden_context])
        context_cell = tf.nn.rnn_cell.GRUCell(n_hidden_context)
        output_context, _ = tf.nn.rnn(context_cell, final_states_enc, initial_state = init_state_con, sequence_length = context_length, scope = con_scope)

    # Context to decoder
    W_con2dec = tf.get_variable('W_con2dec', [n_hidden_context, n_hidden_decoder])
    b_con2dec = tf.get_variable('b_con2dec', [n_hidden_decoder])
    init_state_dec = [tf.tanh(tf.matmul(fs, W_con2dec) + b_con2dec) for fs in output_context]

    # Decoder RNN
    output_dec = _build_decoders(y_data, y_data_length, n_hidden_decoder, init_state_dec, batch_size, emb_dim)

    # To output
    with tf.variable_scope('output'):
        W_out = tf.get_variable('W_out', [n_hidden_decoder, emb_dim])
        E_out = tf.get_variable('E_out', [emb_dim, emb_dim])
        b_out = tf.get_variable('b_out', [emb_dim])

        output_dec = sum(output_dec,[])
        y_data = sum(y_data,[])
        logits = [tf.matmul(o, W_out) + tf.matmul(y, E_out) + b_out for o, y in zip(output_dec, y_data)]
        logits_words = [tf.matmul(l, W_embedding, transpose_b = True) for l in logits] #TODO: not very efficient, implement(?): http://sebastianruder.com/word-embeddings-softmax/
        logits_words = tf.pack(logits_words,axis=1)
        logits_words = tf.reshape(logits_words, [-1, embedding_shape[0]])

    # Calculate loss
    with tf.variable_scope('loss'):
        y_data_input = tf.slice(sequence_input, [0,1,0], [-1,-1,-1])
        y_data_input_flat = tf.reshape(y_data_input, [-1])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_words, y_data_input_flat)
        not_pad_unk_mask = tf.not_equal(y_data_input_flat, tf.constant(0, tf.int32))
        losses = tf.boolean_mask(losses, not_pad_unk_mask) # Not count loss from unks and padding
        total_loss = tf.reduce_mean(losses)

    return vocab_input, sequence_input, length_input, total_loss

def _build_encoders(x_sequences, x_length, n_hidden_enc, batch_size):
    # TODO: bidirectional?
    # x_sequences - list(num_seq - 1 *[batch_size, num_steps, emb_dim])
    with tf.variable_scope('encoder') as enc_scope:
        init_state = tf.zeros([batch_size, n_hidden_enc])
        cell = tf.nn.rnn_cell.GRUCell(n_hidden_enc)
        final_states = []
        for (x_seq, x_len) in zip(x_sequences, x_length):
            x = tf.unpack(x_seq, axis = 1) # list(num_steps * [batch_size, emb_dim])
            _, final_state = tf.nn.rnn(cell, x, initial_state = init_state, sequence_length = x_len, scope = enc_scope)
            tf.get_variable_scope().reuse_variables()
            final_states.append(final_state)
    return final_states

def _build_decoders(y_sequences, y_length, n_hidden_dec, init_states, batch_size, emb_dim):
    # TODO: bidirectional?
    # y_sequences - list(num_seq - 1 *[batch_size, num_steps, emb_dim])
    with tf.variable_scope('decoder') as dec_scope:
        cell = tf.nn.rnn_cell.GRUCell(n_hidden_dec)
        outputs_list = []
        for (y_seq, y_len, init_s) in zip(y_sequences, y_length, init_states):
            outputs, _ = tf.nn.rnn(cell, y_seq, initial_state = init_s, sequence_length = y_len, scope = dec_scope)
            tf.get_variable_scope().reuse_variables()
            outputs_list.append(outputs)
    return outputs_list
