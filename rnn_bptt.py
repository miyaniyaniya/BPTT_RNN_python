import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(0)
tf.set_random_seed(123)



# モデルの出力を定義
def inference(x, n_batch, maxlen=None, n_hidden=None, n_out=None):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01) # Tensorを正規分布かつ標準偏差の２倍までのランダムな値で初期化する
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape, dtype=tf.float32) # TensorのデータタイプとShapeを指定する
        return tf.Variable(initial)

    #ここがtrensorflowのすごいとこで、隠れ層の状態（時系列を）cellが保持している
    #ここの内部構造については、コード読みするなり、理解を深めるのが必要
    cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
    initial_state = cell.zero_state(n_batch, tf.float32)# cellの初期化

    state = initial_state
    outputs = []  # 過去の隠れ層の出力を保存
    with tf.variable_scope('RNN'):# 各層につけた変数名管理のためのtensorflowのスコープ定義
        for t in range(maxlen):
            if t > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(x[:, t, :], state)# 時刻tにおける出力
            outputs.append(cell_output)

    output = outputs[-1]

    V = weight_variable([n_hidden, n_out])# 隠れから出力への重みの初期化
    c = bias_variable([n_out])# 隠れから出力へのバイアスの初期化
    y = tf.matmul(output, V) + c #重要：出力は過去の隠れの出力と重みとバイアスからなる！
    # 順伝播の式

    return y




# 誤差関数の定義
def loss(y, t):
    mse = tf.reduce_mean(tf.square(y - t))# 2乗誤差関数
    return mse





# 学習アルゴリズムの定義:実装しといてあれやけど、アダム法ってなんやねん
def training(loss):
    optimizer = \
        tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
                               # 学習率              減衰率の一般的設定

    train_step = optimizer.minimize(loss)
    return train_step


# 学習の収束
class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience: # ループの最低数10が0以下になると終わり
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False



if __name__ == '__main__':
    def sin(x, T=100): # 予測するデータ
        return np.sin(2.0 * np.pi * x / T)

    def toy_problem(T=100, ampl=0.05): #誤差入りのデータ
        x = np.arange(0, 2 * T + 1)
        noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
        return sin(x) + noise

    ##############
    # データ作成
    ##############
    T = 100
    f = toy_problem(T)

    length_of_sequences = 2 * T  # 全時系列の長さ
    maxlen = 25  # ひとつの時系列データの長さ：どこまで誤差入りデータを与えて、残りを推察させるか
                 # 過去、どれくらいまでさかのぼって学習させるか
                 # これを大きくすればするほど精度が上がるのでは？

    data = []
    target = []

    for i in range(0, length_of_sequences - maxlen + 1):
        data.append(f[i: i + maxlen])
        target.append(f[i + maxlen])

    X = np.array(data).reshape(len(data), maxlen, 1)
    Y = np.array(target).reshape(len(data), 1)

    ##############
    # データ設定
    ##############
    N_train = int(len(data) * 0.9)
    N_validation = len(data) - N_train

    X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X, Y, test_size=N_validation)
    
    ##############
    # モデルの設定
    ##############
    n_in = len(X[0][0])
    n_hidden = 20
    n_out = len(Y[0])

    x = tf.placeholder(tf.float32, shape=[None, maxlen, n_in])
    t = tf.placeholder(tf.float32, shape=[None, n_out])
    n_batch = tf.placeholder(tf.int32)

    y = inference(x, n_batch, maxlen=maxlen, n_hidden=n_hidden, n_out=n_out)
    loss = loss(y, t)
    train_step = training(loss)

    early_stopping = EarlyStopping(patience=10, verbose=1)
    history = {
        'val_loss': []
    }

    ##############
    # モデルの学習
    ##############
    epochs = 500
    batch_size = 10

    init = tf.global_variables_initializer()
    sess = tf.Session()
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

    
    # 出力を用いて予測
    truncate = maxlen
    Z = X[:1]  # 元データの最初の一部だけ切り出し

    original = [f[i] for i in range(maxlen)]
    predicted = [None for i in range(maxlen)]

    for i in range(length_of_sequences - maxlen + 1):
        # 最後の時系列データから未来を予測
        z_ = Z[-1:]
        y_ = y.eval(session=sess, feed_dict={
            x: Z[-1:],
            n_batch: 1
        })
        # 予測結果を用いて新しい時系列データを生成
        sequence_ = np.concatenate(
            (z_.reshape(maxlen, n_in)[1:], y_), axis=0) \
            .reshape(1, maxlen, n_in)
        Z = np.append(Z, sequence_, axis=0)
        predicted.append(y_.reshape(-1))

    
    # グラフで可視化
    plt.rc('font', family='serif')
    plt.figure()
    plt.ylim([-1.5, 1.5])
    plt.plot(toy_problem(T, ampl=0), linestyle='dotted', color='#aaaaaa')
    plt.plot(original, linestyle='dashed', color='black')
    plt.plot(predicted, color='black')
    plt.show()
