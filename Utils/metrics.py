#import tensorflow.keras.backend as K
from keras import backend as K

def recall(_y_true, _y_pred):
    true_positives = K.sum(K.round(K.clip(_y_true * _y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(_y_true, 0, 1)))
    _recall = true_positives / (possible_positives + K.epsilon())
    return _recall


def precision(_y_true, _y_pred):
    true_positives = K.sum(K.round(K.clip(_y_true * _y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(_y_pred, 0, 1)))
    _precision = true_positives / (predicted_positives + K.epsilon())
    return _precision


def f_score(y_true, y_pred, beta: int):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1 + beta**2) * ((p * r) / (beta**2 * p + r + K.epsilon()))


def f1_score(x, y):
    return f_score(x, y, 1)


def f2_score(x, y):
    return f_score(x, y, 2)

