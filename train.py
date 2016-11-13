import numpy as np
import tensorflow as tf
from datetime import datetime
import mr_rnn, data_aux

# Settings
evaluation_step_size = 2

batch_size = 40
num_seq = 6
num_epochs = 1

learning_rate = 0.0002
gradient_clipping = 1.

n_hidden_coarse_prediction = 50
nl_kwargs = {'num_seq' : num_seq,
    'num_steps' : 20,
    'batch_size' : batch_size,
    'n_hidden_encoder' : 50, # for backward and forward cell each
    'n_hidden_context' : 100, 
    'n_hidden_decoder' : 200}
coarse_kwargs = {'num_seq' : num_seq,
    'num_steps' : nl_kwargs['num_steps'] + 1,
    'batch_size' : batch_size,
    'n_hidden_encoder' : 100, # for backward and forward cell each
    'n_hidden_context' : 100,
    'n_hidden_decoder' : 200}

# Load data and create batches
timer = datetime.now()
data = data_aux.generate_full_ubuntu_data_with_coarse(max_utterances = num_seq, max_tokens = nl_kwargs['num_steps'], min_frequency_nl = 10, min_frequency_coarse = 10)
batches = data_aux.batch_iter([data['coarse_data_train'], data['coarse_length_train'], data['nl_data_train'], data['nl_length_train']],batch_size=batch_size, num_epochs=num_epochs)
timer = datetime.now() - timer
print('Data loaded, time spent:', timer)

# Load word embedding
timer = datetime.now()
#coarse_W_embedding = data_aux.pretrained_embedding(data['coarse_vocab_processor'])
#nl_W_embedding = data_aux.pretrained_embedding(data['nl_vocab_processor'])
coarse_W_embedding = data_aux.random_embedding(data['coarse_vocab_processor'], embedding_dim =300)
nl_W_embedding = data_aux.random_embedding(data['nl_vocab_processor'], embedding_dim =300)
coarse_kwargs['embedding_shape'] = coarse_W_embedding.shape
nl_kwargs['embedding_shape'] = nl_W_embedding.shape
timer = datetime.now() - timer
print('Embeddings loaded, time spent:', timer)

# Build graph
timer = datetime.now()
tf.reset_default_graph()
graph_nodes = mr_rnn.build_graph(coarse_kwargs, n_hidden_coarse_prediction, nl_kwargs, learning_rate, gradient_clipping)
timer = datetime.now() - timer
print('Graph built, time spent:', timer)

# Train network
acc_loss = 0
with tf.Session() as sess:
    timer = datetime.now()
    feed_dict={graph_nodes['coarse_vocab_input'] : coarse_W_embedding,
               graph_nodes['nl_vocab_input'] : nl_W_embedding
              }
    sess.run(tf.initialize_all_variables(), feed_dict=feed_dict)
    timer = datetime.now() - timer
    print('Time init graph: ', timer)
    timer = datetime.now()
    acc_loss = 0
    for i, batch in enumerate(batches):
        coarse_seq, coarse_len, nl_seq, nl_len = batch
        feed_dict = {graph_nodes['coarse_sequence_input'] : coarse_seq,
                     graph_nodes['coarse_length_input'] : coarse_len,
                     graph_nodes['nl_sequence_input'] : nl_seq,
                     graph_nodes['nl_length_input'] : nl_len
                     }
        loss, _ = sess.run([graph_nodes['total_loss'], graph_nodes['train_step']], feed_dict=feed_dict)
        print('Train step: %d, loss: %7f, time spent training: %s' % (i, loss, str(datetime.now() - timer)))
        # TODO: evaluate progress with dev set
        # TODO: save model regularly
        # TODO: early stopping
