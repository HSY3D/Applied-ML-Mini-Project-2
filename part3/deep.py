from __future__ import print_function

import numpy as np
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn import cross_validation  
from scipy.sparse import csr_matrix 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import MultinomialNB
from numpy import newaxis

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


data = pd.read_csv('./trainingdata/ml_trainingset.csv',quotechar = '"').as_matrix()
data = data[0:10000,:]
split = data.shape[0]*0.8
split_test = data.shape[0]*0.9


X_train = data[:split,1]
X_val = data[split:split_test,1]
X_test = data[split_test:,1]


y_train = data[:split,2]
y_val = data[split:split_test,2]
y_test = data[split_test:,2]


X_train = X_train.astype(str)
X_val = X_val.astype(str)
X_test = X_test.astype(str)
y_train = y_train.astype(np.uint8)
y_val = y_val.astype(np.uint8)
y_test = y_test.astype(np.uint8)


count_vect = CountVectorizer(stop_words='english')

X_train_counts = count_vect.fit_transform(X_train)
X_val_counts = count_vect.transform(X_val)
X_test_counts = count_vect.transform(X_test)

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)

X_train = np.array(tf_transformer.transform(X_train_counts).todense().astype(np.float32)[:,newaxis,:])
X_val = np.array(tf_transformer.transform(X_val_counts).todense().astype(np.float32)[:,newaxis,:])
X_test = np.array(tf_transformer.transform(X_test_counts).todense().astype(np.float32)[:,newaxis,:])



def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None,1,X_train.shape[2]),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=810,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=310,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.4)
    l_hid3 = lasagne.layers.DenseLayer(
        l_hid2_drop, num_units=110,
        nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid3_drop = lasagne.layers.DropoutLayer(l_hid3, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid3_drop, num_units=4,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


input_var = T.tensor3('inputs')
target_var = T.ivector('targets')
network = build_mlp(input_var)

##pred
prediction = lasagne.layers.get_output(network)

###loss 
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

## parameter update
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)


### loss for test (without the dropout)
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
test_loss = test_loss.mean()

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, updates=updates)

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

# Finally, launch the training loop.
num_epochs = 250
print("Starting training...")
    # We iterate over epochs:
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:

    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
        #print int(batch)
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))
