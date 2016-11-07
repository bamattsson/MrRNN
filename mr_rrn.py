import numpy as np
import tensorflow as tf

def build_graph(embedding_shape, num_seq, num_steps, batch_size, n_hidden_encoder, n_hidden_context, n_hidden_decoder, learning_rate):
    """Constructs the graph."""
    # Input
    # TODO: variable batch_size
    sequence_input = tf.placeholder(tf.int32, [batch_size, num_seq, num_steps])
    
    # Get embedding
    with tf.device('/cpu:0'), tf.name_scope('embedding'):
        # TODO: paper wants different i/o embeddings?
        vocab_input = tf.placeholder(tf.float32, shape = embedding_shape)
        emb_dim = embedding_shape[1]
        W_embedding = tf.Variable(vocab_input, name='W') # [vocab_size, emb_dim]
        sequence = tf.nn.embedding_lookup(W_embedding, sequence_input) # [batch_size, num_seq, num_steps, emb_dim]

        sequence = tf.unpack(sequence, axis=1) # list(num_seq * [batch_size, num_steps, emb_dim])
        x = sequence[:-1] # list(num_seq - 1 *[batch_size, num_steps, emb_dim])
        y_input = sequence[1:] # list(num_seq - 1 *[batch_size, num_steps, emb_dim])
        y_input = [tf.unpack(y_seq, axis = 1) for y_seq in y_input] # packing up time steps
        y_input = [[tf.zeros([batch_size, emb_dim])] + y_seq[:-1] for y_seq in y_input] # adding initial time step and removing last
        # y_input: list(num_seq - 1 *list(num_steps * [batch_size, emd_dim]))

    # Encoder RNN
    final_states_enc = _build_encoders(x, n_hidden_encoder, batch_size)

    # Context RNN
    with tf.variable_scope('context') as con_scope:
        init_state_con = tf.zeros([batch_size, n_hidden_context])
        context_cell = tf.nn.rnn_cell.GRUCell(n_hidden_context)
        output_context, _ = tf.nn.rnn(context_cell, final_states_enc, initial_state = init_state_con, scope = con_scope)

    # Context to decoder
    W_con2dec = tf.get_variable('W_con2dec', [n_hidden_context, n_hidden_decoder])
    b_con2dec = tf.get_variable('b_con2dec', [n_hidden_decoder])
    init_state_dec = [tf.tanh(tf.matmul(fs, W_con2dec) + b_con2dec) for fs in output_context]

    # Decoder RNN
    output_dec = _build_decoders(y_input, n_hidden_decoder, init_state_dec, batch_size, emb_dim)

    # To output
    with tf.variable_scope('output'):
        W_out = tf.get_variable('W_out', [n_hidden_decoder, emb_dim])
        E_out = tf.get_variable('E_out', [emb_dim, emb_dim])
        b_out = tf.get_variable('b_out', [emb_dim])

        output_dec = sum(output_dec,[])
        y_input = sum(y_input,[])
        logits = [tf.matmul(o, W_out) + tf.matmul(y, E_out) + b_out for o, y in zip(output_dec, y_input)]
        logits_words = [tf.matmul(l, W_embedding, transpose_b = True) for l in logits] #TODO: not very efficient, implement(?): http://sebastianruder.com/word-embeddings-softmax/
        logits_words = tf.pack(logits_words,axis=1)
        logits_words = tf.reshape(logits_words, [-1, embedding_shape[0]])
    
    # Calculate loss
    with tf.variable_scope('Loss'):
        y_output = tf.slice(sequence_input, [0, 1, 0], [-1, -1, -1])
        y_output = tf.reshape(y_output, [-1])

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_words, y_output)
        total_loss = tf.reduce_mean(losses)

    # Do training
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss) 

    return (vocab_input, sequence_input, total_loss, train_step)

def _build_encoders(x_sequences, n_hidden_enc, batch_size):
    # TODO: bidirectional?
    # x_sequences - list(num_seq - 1 *[batch_size, num_steps, emb_dim])
    with tf.variable_scope('encoder') as enc_scope:
        init_state = tf.zeros([batch_size, n_hidden_enc])
        cell = tf.nn.rnn_cell.GRUCell(n_hidden_enc)
        final_states = []
        for x_seq in x_sequences:
            x = tf.unpack(x_seq, axis = 1) # list(num_steps * [batch_size, emb_dim])
            _, final_state = tf.nn.rnn(cell, x, initial_state = init_state, scope = enc_scope)
            tf.get_variable_scope().reuse_variables()
            final_states.append(final_state)
    return final_states

def _build_decoders(y_sequences, n_hidden_dec, init_states, batch_size, emb_dim):
    # TODO: bidirectional?
    # y_sequences - list(num_seq - 1 *[batch_size, num_steps, emb_dim])
    with tf.variable_scope('decoder') as dec_scope:
        cell = tf.nn.rnn_cell.GRUCell(n_hidden_dec)
        outputs_list = []
        for (y_seq, init_s) in zip(y_sequences, init_states):
            outputs, _ = tf.nn.rnn(cell, y_seq, initial_state = init_s, scope = dec_scope)
            tf.get_variable_scope().reuse_variables()
            outputs_list.append(outputs)
    return outputs_list
