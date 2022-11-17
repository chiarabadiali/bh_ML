import numpy as np
from matplotlib import pyplot as plt

def prepare_data(dataset):

    # Get the dimensions
    M, N = dataset.shape

    # Shuffle the data
    np.random.seed(42)
    np.random.shuffle(dataset)
    iX = np.arange(dataset.shape[0])
    np.random.shuffle(iX)
    dataset = dataset[iX]

    # Assign data and target
    data = dataset[:,0:N-1]
    targets = dataset[:,N-1]

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

    # Normalisation of target
    norm = 1.49e26
    dataset[:,N-1] *= norm

    if np.isnan(np.min(targets)) == False:
        return train_data, train_targets, val_data, val_targets, test_data, test_targets

def plot_histo(x, y, bins,logscale):
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
    y = class_data[:,2]
    n, edges, _ = plt.hist(y, nbins, color = 'indianred', alpha=0.5, label='Osiris')
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

    return np.array(data)

def balance_data2(class_data, nbins, ratio):
    y = class_data[:,2]
    n, edges, _ = plt.hist(y, nbins, color = 'indianred', alpha=0.5, label='Osiris')
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

def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['mae']
    val_accuracy = history.history['val_mae']


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