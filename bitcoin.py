import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import pandas as pd
import datetime
import os

import matplotlib.dates as mdates

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

def inference(x, n_batch, maxlen=None, n_hidden=None, n_out=None):
    def weight_variable(shape):
        initial = np.sqrt(2.0 / shape[0]) * tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial)

    # LSTMをここで利用する
    cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    initial_state = cell.zero_state(n_batch, tf.float32)

    state = initial_state
    outputs = []  # 過去の隠れ層の出力を保存

    # LSTMの名前空間をつける
    with tf.variable_scope('LSTM'):
        for t in range(maxlen):
            if t > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(x[:, t, :], state)
            outputs.append(cell_output)

    output = outputs[-1]

    V = weight_variable([n_hidden, n_out])
    c = bias_variable([n_out])
    y = tf.matmul(output, V) + c  # 線形活性

    return y


def loss(y, t):
    mse = tf.reduce_mean(tf.square(y - t))

    return mse


def training(loss):
    optimizer = \
        tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)

    train_step = optimizer.minimize(loss)
    return train_step

def make_elapsed_dates(first, last):

    elapsed_date = []
    elapsed_days = []

    first_date = datetime.datetime.strptime(first, '%Y-%m-%d %H:%M:%S')
    last_date = datetime.datetime.strptime(last, '%Y-%m-%d %H:%M:%S')
    while first_date <= last_date:
        elapsed_days.append((first_date - datetime.datetime(2009, 1, 1)).days / 604)
        elapsed_date.append(first_date)
        first_date += datetime.timedelta(days = 2)

    length_of_sequences = len(elapsed_days)
    maxlen = 25

    elapsed_days_data = []
    for i in range(0, length_of_sequences - maxlen):
        elapsed_days_data.append(elapsed_days[i: i + maxlen])

    X = np.array(elapsed_days_data).reshape(len(elapsed_days_data), maxlen, 1)

    return [X, elapsed_date]

class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False


if __name__ == '__main__':
    def sin(x, T=100):
        return np.sin(2.0 * np.pi * x / T)

    def toy_problem(T=100, ampl=0.05):
        x = np.arange(0, 2 * T + 1)
        noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
        return sin(x) + noise

    '''
    データの生成
    '''
    marketPrices = []
    for date, price in pd.read_csv('market-price.csv').values.tolist():
        marketPrices.append([
            # 日付を2009/1/1からの経過日数にして、正規化する
            (datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S") - datetime.datetime(2009, 1, 1)).days / 604,
            price
        ])
    marketPrices = pd.DataFrame(data=marketPrices).values

    length_of_sequences = len(marketPrices)
    maxlen = 25

    data = []
    target = []

    for i in range(0, length_of_sequences - maxlen):
        data.append(marketPrices[i: i + maxlen, 0])
        target.append(marketPrices[i + maxlen][1])

    X = np.array(data).reshape(len(data), maxlen, 1)
    Y = np.array(target).reshape(len(data), 1)

    # データ設定
    N_train = int(len(data) * 0.9)
    N_validation = len(data) - N_train

    X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X, Y, test_size=N_validation)

    '''
    モデル設定
    '''
    n_in = len(X[0][0])  # 1
    n_hidden = 300
    n_out = len(Y[0])  # 1

    x = tf.placeholder(tf.float32, shape=[None, maxlen, n_in])
    t = tf.placeholder(tf.float32, shape=[None, n_out])
    n_batch = tf.placeholder(tf.int32, shape=[])

    y = inference(x, n_batch, maxlen=maxlen, n_hidden=n_hidden, n_out=n_out)
    loss = loss(y, t)
    train_step = training(loss)

    early_stopping = EarlyStopping(patience=300, verbose=1)
    history = {
        'val_loss': []
    }

    '''
    モデル学習
    '''
    epochs = 2000
    batch_size = 10

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, MODEL_DIR + '/model.ckpt')

    '''
    学習せずに、保存したモデルを使う
    
    sess.run(init)
    
    n_batches = N_train // batch_size

    for epoch in range(epochs):
        X_, Y_ = shuffle(X_train, Y_train)

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            sess.run(train_step, feed_dict={
                x: X_[start:end],
                t: Y_[start:end],
                n_batch: batch_size
            })

        # 検証データを用いた評価
        val_loss = loss.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            n_batch: N_validation
        })

        history['val_loss'].append(val_loss)
        print('epoch:', epoch,
              ' validation loss:', val_loss)

        # Early Stopping チェック
        if early_stopping.validate(val_loss):
            break

    # 学習したモデルを保存する
    model_path = saver.save(sess, MODEL_DIR + '/model.ckpt')
    print('Model saved to:', model_path)
    '''

    '''
    出力を用いて予測
    '''
    (X, elapsed_date) = make_elapsed_dates('2009-01-03 00:00:00', '2020-01-02 00:00:00')
    elapsed_date = elapsed_date[:(len(elapsed_date) - maxlen - 1)]

    predicted = [None for i in range(maxlen)]

    for i in range(len(X) - maxlen - 1):
        start = i * batch_size
        end = start + batch_size
        y_ = y.eval(session=sess, feed_dict={
            x: X[i:i+1],
            n_batch: 1
        })
        predicted.append(y_.reshape(-1))


    '''
    グラフで可視化
    '''
    ax = plt.subplot()
    ax.plot(elapsed_date, predicted, color='black')
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=130))
    ax.set_xlim(datetime.datetime(2009, 1, 3), datetime.datetime(2020, 1, 2))
    plt.plot(elapsed_date[:1825], marketPrices[:, 1], linestyle='dotted', color='#aaaaaa')
    plt.xticks(rotation=70)
    plt.show()