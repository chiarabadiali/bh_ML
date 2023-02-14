import numpy as np
from matplotlib import pyplot as plt
from keras import models
from keras import layers
from tensorflow.keras.layers import BatchNormalization
import progressbar
import struct

def set_seed():

    # This routine enables to set all the seeds to get reproducible results
    # with your python script on one computer's/laptop's CPU

    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value = 42

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.random.set_seed(seed_value)
    # for later versions: 
    # tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    from keras import backend as K
    # for later versions:
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

def build_model_tcs(n_layers=2, n_nodes=4):
    model = models.Sequential()
    model.add (BatchNormalization(input_dim = 2))
    for i in range(n_layers):
        model.add (layers.Dense(n_nodes, activation="relu"))
    model.add (layers.Dense(1, activation="relu"))
    model.compile(optimizer = "rmsprop", loss='mse', metrics=["mape", "mse"])
    return model

def build_model_cdf(n_layers=2, n_nodes=4):
    model = models.Sequential()
    model.add (BatchNormalization(input_dim = 3))
    for i in range(n_layers):
        model.add (layers.Dense(n_nodes, activation="relu"))
    model.add (layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer = "rmsprop", loss='binary_crossentropy', metrics=['accuracy'])
    return model

def prepare_targets_tcs(ds):
    _ds = ds.copy()
    M, N = _ds.shape

    _ds[:,0] = np.log10(_ds[:,0])
    _ds[:,1] = np.log10(_ds[:,1])

    _ds[:,N-1] = -np.log10(_ds[:,N-1])
    _ds[:,N-1] = (_ds[:,N-1]-min(_ds[:,N-1])) / (max(_ds[:,N-1])-min(_ds[:,N-1]))

    return _ds

def prepare_targets_cdf(ds):
    _ds = ds.copy()
    M, N = _ds.shape

    _ds[:,0] = np.log10(_ds[:,0])
    _ds[:,1] = np.log10(_ds[:,1])
    _ds[:,3] = _ds[:,3] - _ds[:,2]
    _ds[:,3] = (_ds[:,3]-min(_ds[:,3])) / (max(_ds[:,3])-min(_ds[:,3]))

    return _ds

def train_test_split(ds):

    # Get the dimensions
    _ds = ds.copy()
    M, N = _ds.shape

    # Shuffle the data
    np.random.shuffle(_ds)
    iX = np.arange(_ds.shape[0])
    np.random.shuffle(iX)
    _ds = _ds[iX]

    # Assign data and target
    data = _ds[:,0:N-1]
    targets = _ds[:,N-1]

    # Split train, validation, test (70%,15%,15%)
    train_split, val_split = 0.70, 0.85
    id1 = int(M*train_split)
    id2 = int(M*val_split)
    print("Train / Val / Test set: {0} / {1} / {2}".format(id1, id2-id1, M-id2))

    # data
    train_data = data[:id1]
    val_data = data[id1:id2]
    test_data = data[id2:]

    # target
    train_targets = targets[:id1]
    val_targets = targets[id1:id2]
    test_targets = targets[id2:]

    if np.isnan(np.min(targets)) == False:
        return train_data, train_targets, val_data, val_targets, test_data, test_targets

def plot_histo(y, bins,logscale):
    y = np.array(y)
    plt.hist(y, bins, color = 'indianred', alpha=0.5, label='Osiris')
    plt.legend(loc='upper right')
    plt.xlabel('target')
    plt.ylabel('number of occurrences')
    if logscale == 1:
        plt.yscale('log')
    plt.show()

def load_data(name):
    return np.loadtxt(name, delimiter=',')

def balance_data(class_data, nbins):
    y = class_data[:,-1]
    n, edges, _ = plt.hist(y, nbins, color = 'indianred', alpha=0.5, label='original')
    plt.close()
    n_max = int(n.max())
    data = []
    bar = progressbar.ProgressBar(maxval=len(class_data), 
                              widgets=[progressbar.Bar('=', '[', ']'), ' ', 
                                       progressbar.Percentage(), 
                                       " of {0}".format(len(class_data))])
    bar.start()
    for k, class_ in enumerate(class_data):
        for i in range(len(n)):
            edges_min = edges[i]
            edges_max = edges[i+1]
            if class_[2] >= edges_min and class_[2] <= edges_max:
                for j in range(int((n_max/n[i]))):
                    data.append(class_)
                break
        bar.update(k+1)
    bar.finish()

    data = np.array([data])
    data = data[0,:,:]

    return data

def balance_data2(class_data, nbins, ratio):
    y = class_data[:,-1]
    n, edges, _ = plt.hist(y, nbins, color = 'indianred', alpha=0.5, label='Osiris')
    plt.close()
    n_max = int(n.max())*ratio
    data = []
    bar = progressbar.ProgressBar(maxval=len(class_data), 
                              widgets=[progressbar.Bar('=', '[', ']'), ' ', 
                                       progressbar.Percentage(), 
                                       " of {0}".format(len(class_data))])
    bar.start()
    cow = 0
    for k, class_ in enumerate(class_data):
        for i in range(len(n)):
            edges_min = edges[i]
            edges_max = edges[i+1]
            if class_[0] > edges_min and class_[0] < edges_max:
                if int(n[i]/n_max) > 1 :
                    step = int(n[i]/n_max)
                    if cow%step==0:
                        data.append(class_)
                    cow +=1
                else:
                    for j in range(int(n_max/(n[i]))):
                        data.append(class_)
        bar.update(k+1)
            
    bar.finish()

    #plt.hist(data, nbins, color = 'indianred', alpha=0.5, label='Osiris')
    
    return np.array(data)

def plot_history_tcs(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['mape']
    val_accuracy = history.history['val_mape']


    epochs = range(1, len(loss) + 1)
    fig, ax1 = plt.subplots()

    l1 = ax1.plot(epochs, loss, 'bo', label='Training loss')
    vl1 = ax1.plot(epochs, val_loss, 'b', label='Validation loss')
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (mape)')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ac2= ax2.plot(epochs, accuracy, 'o', c="red", label='Training acc')
    vac2= ax2.plot(epochs, val_accuracy, 'r', label='Validation acc')
    ax2.set_ylabel('mape')
    ax2.set_yscale('log')

    lns = l1 + vl1 + vac2 + ac2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc="center right")
    fig.tight_layout()
    fig.show()

def plot_history_cdf(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']


    epochs = range(1, len(loss) + 1)
    fig, ax1 = plt.subplots()

    l1 = ax1.plot(epochs, loss, 'bo', label='Training loss')
    vl1 = ax1.plot(epochs, val_loss, 'b', label='Validation loss')
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (mape)')
#    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ac2= ax2.plot(epochs, accuracy, 'o', c="red", label='Training acc')
    vac2= ax2.plot(epochs, val_accuracy, 'r', label='Validation acc')
    ax2.set_ylabel('accuracy')
#    ax2.set_yscale('log')

    lns = l1 + vl1 + vac2 + ac2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc="center right")
    fig.tight_layout()
    fig.show()


def balance_data3(dataset, nbins):

    M, N = dataset.shape
    y = dataset[:,-1]
    dy = (max(y) - min(y)) / y.shape
    edges = np.linspace(min(y)-dy/2., max(y)+dy/2., nbins+1)

    _ds = list()
    for i in range(nbins):
        indexes = np.array([ j for j in range(M) if (y[j]>=edges[i]) and (y[j]<edges[i+1]) ])
        indexes_balanced = np.random.choice(indexes, int(M/nbins))
        for index in indexes_balanced:
           _ds.append(dataset[index,:])

    return np.array(_ds)