import numpy as np
import tensorflow as tf

def build_graph(num_seq, num_steps, batch_size, n_hidden_encoder, n_hidden_context, n_hidden_decoder, learning_rate):
    """Constructs the graph."""
    # Input
    sequence_input = tf.placeholder(tf.int32, [batch_size, num_seq, num_steps])

    sequence = tf.one_hot(sequence_input, 3) # [batch_size, num_seq, num_steps, 1-hot]
    sequence = tf.unpack(sequence, axis=1) # list(num_seq * [batch_size, num_steps, 1-hot])
    x = sequence[:-1] # list(num_seq - 1 *[batch_size, num_steps, 1-hot]) 
    y = sequence[1:] # list(num_seq - 1 *[batch_size, num_steps, 1-hot])

    # Encoder RNN
    init_state_enc = tf.zeros([batch_size, n_hidden_encoder])
    final_states_enc = _build_encoders(x, n_hidden_encoder, init_state_enc)

    # Context RNN
    init_state_con = tf.zeros([batch_size, n_hidden_context])
    context_cell = tf.nn.rnn_cell.GRUCell(n_hidden_context)
    output_context, _ = tf.nn.rnn(context_cell, final_states_enc, initial_state = init_state_con)

    # Context to decoder
    W_con_to_dec = tf.get_variable('W_con_to_dec', [n_hidden_context, n_hidden_decoder])
    init_state_dec = [tf.matmul(fs, W_con_to_dec) for fs in output_context]

    # Decoder RNN
    output_dec = _build_decoders(y, n_hidden_decoder, init_state_dec, batch_size)
    output_dec = sum(output_dec,[])

    # To output
    W_out = tf.get_variable('W_out', [n_hidden_decoder, 3])
    b_out = tf.get_variable('b_out', [3])

    logits = [tf.matmul(o, W_out) + b_out for o in output_dec]
    predictions = [tf.nn.softmax(l) for l in logits]
    y = sum([tf.unpack(yv, axis = 1) for yv in y],[])

    y = [tf.argmax(yv, 1) for yv in y]
    loss_weights = [tf.ones([batch_size]) for i in range(num_steps*(num_seq- 1))]

    # Cost and training
    losses = tf.nn.seq2seq.sequence_loss_by_example(logits, y, loss_weights)
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss) 

    return (sequence_input, total_loss, train_step)

def _build_encoders(x_sequences, n_hidden_enc, init_state):
    # x_sequences - list(num_seq - 1 *[batch_size, num_steps, 1-hot])
    with tf.variable_scope('encoder') as enc_scope:
        cell = tf.nn.rnn_cell.GRUCell(n_hidden_enc)
        final_states = []
        for x_seq in x_sequences:
            x = tf.unpack(x_seq, axis = 1) # list(num_steps * [batch_size, 1-hot])
            _, final_state = tf.nn.rnn(cell, x, initial_state = init_state, scope = enc_scope)
            tf.get_variable_scope().reuse_variables()
            final_states.append(final_state)
    return final_states

def _build_decoders(y_sequences, n_hidden_dec, init_states, batch_size):
    # y_sequences - list(num_seq - 1 *[batch_size, num_steps, 1-hot])
    with tf.variable_scope('decoder') as dec_scope:
        cell = tf.nn.rnn_cell.GRUCell(n_hidden_dec)
        outputs_list = []
        for (y_seq, init_s) in zip(y_sequences, init_states):
            y = tf.unpack(y_seq, axis = 1) # list(num_steps * [batch_size, 1-hot])
            y = [tf.zeros([batch_size, 3])] + y 
            outputs, _ = tf.nn.seq2seq.rnn_decoder(y[:-1], init_s, cell, scope = dec_scope)
            tf.get_variable_scope().reuse_variables()
            outputs_list.append(outputs)
    return outputs_list
