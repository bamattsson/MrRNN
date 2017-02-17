## Introduction
A bare-bones but working tensorflow implementation of the paper [Multiresolution Recurrent Neural Networks: An Application to Dialogue Response Generation](https://arxiv.org/abs/1606.00776). Done during 10 days as a task when applying for a job.

## Description of the files
* mr_rnn.py -- builds the graph
* train.py -- to train the model
* data_aux.py -- help functions to generate data
* NB_tests.ipynb -- IPython notebook where the test sequence can be run

## Comments
Built in Python 3.5.2 with Tensorflow 0.10.0.

To run the program you have to add Googles w2v (from https://code.google.com/archive/p/word2vec/) to data/ or use random_embedding instead as well as add the extracted Ubuntu dialogoue corpus (from www.iulianserban.com/Files/UbuntuDialogueCorpus.zip) to data/. 

Note that all number of hidden units (except the word embeddings) are set to 10 % of what is mentioned in the article, this since the RAM of my personal computer wasn't big enough to handle the full parameter sizes. The settings can be changed in train.py.

I have included comments starting with TODO at multiple places in the code followed by thoughts on the design. These are places where I suggest one pays more attention and potentially does alterations if one were to use this model.
