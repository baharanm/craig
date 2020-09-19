from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import numpy as np
import time
from keras.regularizers import l2
import util

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

num_classes, smtk = 10, 0
Y_train_nocat = Y_train
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)

batch_size = 32
# subset, random = False, False  # all
subset, random = True, False  # greedy
# subset, random = True, True  # random
subset_size = .4 if subset else 1.0
epochs = 15
reg = 1e-4
runs = 5
save_subset = False

# path to save the results
folder = f'/tmp/mnist'


model = Sequential()
model.add(Dense(100, input_dim=784, kernel_regularizer=l2(reg)))
model.add(Activation('sigmoid'))
model.add(Dense(10, kernel_regularizer=l2(reg)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')


train_loss, test_loss = np.zeros((runs, epochs)), np.zeros((runs, epochs))
train_acc, test_acc = np.zeros((runs, epochs)), np.zeros((runs, epochs))
train_time = np.zeros((runs, epochs))
grd_time, sim_time, pred_time = np.zeros((runs, epochs)), np.zeros((runs, epochs)), np.zeros((runs, epochs))
not_selected = np.zeros((runs, epochs))
times_selected = np.zeros((runs, len(X_train)))
best_acc = 0
print(f'----------- smtk: {smtk} ------------')

if save_subset:
    B = int(subset_size * len(X_train))
    selected_ndx = np.zeros((runs, epochs, B))
    selected_wgt = np.zeros((runs, epochs, B))

for run in range(runs):
    X_subset = X_train
    Y_subset = Y_train
    W_subset = np.ones(len(X_subset))
    ordering_time,similarity_time, pre_time = 0, 0, 0
    loss_vec, acc_vec, time_vec = [], [], []
    for epoch in range(0, epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        num_batches = int(np.ceil(X_subset.shape[0] / float(batch_size)))

        for index in range(num_batches):
            X_batch = X_subset[index * batch_size:(index + 1) * batch_size]
            Y_batch = Y_subset[index * batch_size:(index + 1) * batch_size]
            W_batch = W_subset[index * batch_size:(index + 1) * batch_size]

            start = time.time()
            history = model.train_on_batch(X_batch, Y_batch, sample_weight=W_batch)
            train_time[run][epoch] += time.time() - start

        if subset:
            if random:
                # indices = np.random.randint(0, len(X_train), int(subset_size * len(X_train)))
                indices = np.arange(0, len(X_train))
                np.random.shuffle(indices)
                indices = indices[:int(subset_size * len(X_train))]
                W_subset = np.ones(len(indices))
            else:
                start = time.time()
                _logits = model.predict_proba(X_train)
                pre_time = time.time() - start
                features = _logits - Y_train

                indices, W_subset, _, _, ordering_time, similarity_time = util.get_orders_and_weights(
                    int(subset_size * len(X_train)), features, 'euclidean', smtk, 0, False, Y_train_nocat)

                W_subset = W_subset / np.sum(W_subset) * len(W_subset)  # todo

            if save_subset:
                selected_ndx[run, epoch], selected_wgt[run, epoch] = indices, W_subset

            grd_time[run, epoch], sim_time[run, epoch], pred_time[run, epoch] = ordering_time, similarity_time, pre_time
            times_selected[run][indices] += 1
            not_selected[run, epoch] = np.sum(times_selected[run] == 0) / len(times_selected[run]) * 100
        else:
            pred_time = 0
            indices = np.arange(len(X_train))

        X_subset = X_train[indices, :]
        Y_subset = Y_train[indices]

        start = time.time()
        score = model.evaluate(X_test, Y_test, verbose=1)
        eval_time = time.time()-start

        start = time.time()
        score_loss = model.evaluate(X_train, Y_train, verbose=1)
        print(f'eval time on training: {time.time()-start}')

        test_loss[run][epoch], test_acc[run][epoch] = score[0], score[1]
        train_loss[run][epoch], train_acc[run][epoch] = score_loss[0], score_loss[1]
        best_acc = max(test_acc[run][epoch], best_acc)

        grd = 'random_wor' if random else 'grd_normw'
        print(f'run: {run}, {grd}, subset_size: {subset_size}, epoch: {epoch}, test_acc: {test_acc[run][epoch]}, '
              f'loss: {train_loss[run][epoch]}, best_prec1_gb: {best_acc}, not selected %:{not_selected[run][epoch]}')

    if save_subset:
        print(
            f'Saving the results to {folder}_{subset_size}_{grd}_{runs}')

        np.savez(f'{folder}_{subset_size}_{grd}_{runs}',
                 train_loss=train_loss, test_acc=test_acc, train_acc=train_acc, test_loss=test_loss,
                 train_time=train_time, grd_time=grd_time, sim_time=sim_time, pred_time=pred_time,
                 not_selected=not_selected, times_selected=times_selected,
                 subset=selected_ndx, weights=selected_wgt)
    else:
        print(
            f'Saving the results to {folder}_{subset_size}_{grd}_{runs}')

        np.savez(f'{folder}_{subset_size}_{grd}_{runs}',
                 train_loss=train_loss, test_acc=test_acc, train_acc=train_acc, test_loss=test_loss,
                 train_time=train_time, grd_time=grd_time, sim_time=sim_time, pred_time=pred_time,
                 not_selected=not_selected, times_selected=times_selected)

