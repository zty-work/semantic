import json
from keras.models import Model
from keras.layers import *
from keras.constraints import unit_norm
from keras.callbacks import Callback
import keras.backend as K
import numpy as np

infile1 = open("./train_data/classes.txt", "r")
infile2 = open("./train_data/testclasses.txt", "r")


def sparse_logits_categorical_crossentropy(y_true, y_pred, scale=30):
    return K.sparse_categorical_crossentropy(y_true, scale * y_pred, from_logits=True)


def sparse_amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_true = K.expand_dims(y_true[:, 0], 1)
    y_true = K.cast(y_true, 'int32')
    batch_idxs = K.arange(0, K.shape(y_true)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, y_true], 1)
    y_true_pred = K.tf.gather_nd(y_pred, idxs)
    y_true_pred = K.expand_dims(y_true_pred, 1)
    y_true_pred_margin = y_true_pred - margin
    _Z = K.concatenate([y_pred, y_true_pred_margin], 1)
    _Z = _Z * scale
    logZ = K.logsumexp(_Z, 1, keepdims=True)
    logZ = logZ + K.log(1 - K.exp(scale * y_true_pred - logZ))
    return - y_true_pred_margin * scale + logZ


def sparse_simpler_asoftmax_loss(y_true, y_pred, scale=30):
    y_true = K.expand_dims(y_true[:, 0], 1)
    y_true = K.cast(y_true, 'int32')
    batch_idxs = K.arange(0, K.shape(y_true)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, y_true], 1)
    y_true_pred = K.tf.gather_nd(y_pred, idxs)
    y_true_pred = K.expand_dims(y_true_pred, 1)
    y_true_pred_margin = 1 - 8 * K.square(y_true_pred) + 8 * K.square(K.square(y_true_pred))
    y_true_pred_margin = y_true_pred_margin - K.relu(y_true_pred_margin - y_true_pred)
    _Z = K.concatenate([y_pred, y_true_pred_margin], 1)
    _Z = _Z * scale
    logZ = K.logsumexp(_Z, 1, keepdims=True)
    return - y_true_pred_margin * scale + logZ


num_train_groups = 4500
batch_size = 1
word_size = 22
epochs = 10

x_train = []
y_train = []
x_valid = []
y_valid = []
valid_index = []
i = 0
m = 0
while 1:
    line1 = infile1.readline()
    if not line1:
        break
    pass
    if i < num_train_groups:
        tempLine = line1.split(",")
        tempx = tempLine[1:]
        for j in range(0, len(tempx)):
            if tempx[j] == ' 607':
                tempx[j] = 0
            tempx[j] = int(tempx[j])
        tempy = int(tempLine[0])
        x_train.append(tempx)
        y_train.append(tempy)

#    else:
#        tempLine = line1.split(",")
#        tempx = tempLine[1:]
#        for j in range(0, len(tempx)):
#            if tempx[j] == ' 607':
#                tempx[j] = 0
#            tempx[j] = int(tempx[j])
#        tempy = int(tempLine[0])
#        x_valid.append(tempx)
#        y_valid.append(tempy)
#        valid_index.append(m)
#        m = m + 1

while 1:
    line1 = infile2.readline()
    if not line1:
        break
    pass
    tempLine = line1.split(",")
    tempx = tempLine[1:23]
    for j in range(0, 22):
        if tempx[j] == ' 607':
            tempx[j] = 0
        tempx[j] = int(tempx[j])
    tempy = int(tempLine[0])
    x_valid.append(tempx)
    y_valid.append(tempy)

print("x-v-s")
print(np.shape(x_valid))

x_in = Input(shape=(22,))
x_embedded = Embedding(2000,
                       word_size)(x_in)
x = CuDNNGRU(word_size)(x_embedded)

pred = Dense(num_train_groups,
             use_bias=False,
             kernel_constraint=unit_norm())(x)
encoder = Model(x_in, x)  # get encoder
model = Model(x_in, pred)  # train as classifier
model.compile(loss=sparse_amsoftmax_loss,
              optimizer='adam',
              metrics=['sparse_categorical_accuracy'])

# modifying
x_in = Input(shape=(word_size,))
# cal similarity
x = Dense(len(x_valid), use_bias=False)(x_in)
model_sort = Model(x_in, x)


# to see whether two sentences are the same
# the valid set is the test set
def evaluate():
    print ('validing...')
    acc = 0.5 * len(x_valid)
    valid_vec = encoder.predict(np.array(x_valid),
                                verbose=True,
                                batch_size=1000)
    print("valid_vec")
    model_sort.set_weights([valid_vec.T])
    sorted_result = model_sort.predict(valid_vec,
                                       verbose=True,
                                       batch_size=1000)
    print("len", len(sorted_result))
    for j in range(1, len(y_valid), 2):
        if (y_valid[j] == 1) != (sorted_result[j][j - 1] >= 0.6):
            acc = acc - 1
    acc = 2 * acc / (len(x_valid))
    return acc


class Evaluate(Callback):
    def __init__(self):
        self.acc = 0
        self.highest = 0.

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate()
        self.acc = acc
        if acc >= self.highest:
            self.highest = acc
            model.save_weights('sent_sim_amsoftmax.model')
        json.dump({'accs': self.acc, 'highest_top1': self.highest},
                  open('valid_amsoftmax.log', 'w'), indent=4)
        print("acc", self.highest)


evaluator = Evaluate()

history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[evaluator])

# model.save
# model loading
