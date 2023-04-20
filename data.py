import itertools
import numpy as np
import tensorflow as tf

w2v_docs = [
    # word
    "q w e r t y u i o p",
    "a s d f g h j k l z",
    "x c v b n m w s f h",
    "q w e r s f g b n y",
    "p i k h n l o u w q",
    "q w e r t y i g s v",
    "w m l k j a z x o p",
    "p i j h f s l k m n",
    # num
    "1 2 3 4 5 6 7 8 9 6",
    "9 8 7 6 5 4 3 2 1 4",
    "8 6 3 2 1 4 8 9 5 2",
    "1 5 6 9 8 7 3 1 2 4",
    "8 9 8 5 6 3 2 1 4 7",
    "6 5 3 2 1 8 9 7 4 3",
    "8 9 6 3 2 5 7 4 1 3",
    "8 9 8 7 5 5 6 3 2 1"
]

def load_w2v(n_window=2, method="skip_gram"):
    words = [v.split(" ") for v in w2v_docs]
    vocab = list(set(itertools.chain(*words)))   # get vocab
    i2v = {i: v for i, v in enumerate(vocab)}   # index to vocab
    v2i = {v: i for i, v in i2v.items()}   # vocab to index

    js_window = [i for i in range(-n_window,n_window+1) if i != 0]   # window
    x = []
    y = []
    for i in range(len(words)):
        if method == "cbow":
            for j in range(n_window,len(words[i])-n_window):
                temp = []
                for w in js_window:
                    temp.append(v2i[words[i][j+w]])   # get x
                temp.append(v2i[words[i][j]])   # get y
                x.append(temp[:-1])
                y.append(temp[-1])
        if method == "skip_gram":
            for j in range(len(words[i])):
                for w in js_window:
                    if j + w >= 0 and j + w < len(words[i]):
                        x.append([v2i[words[i][j]]])
                        y.append(v2i[words[i][j+w]])

    x = np.array(x)
    y = np.array(y)
    print("x example:")
    print(x[:5,:])
    print("y example:")
    print(y[:5])

    db = tf.data.Dataset.from_tensor_slices((x,y))
    db = db.shuffle(94).batch(8)
    return db, len(v2i), i2v

def load_imdb(vocab_num, maxlen, batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_num)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print("x example:")
    print(x_train[:3,:])
    print("y example:")
    print(y_train[:3])

    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db_train = db_train.shuffle(25000).batch(batch_size)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.shuffle(25000).batch(batch_size)
    return db_train, db_test

if __name__ == "__main__":
    db_train, db_test = load_imdb(vocab_num=10000, maxlen=100, batch_size=64)
    print(next(iter(db_train))[0].shape, next(iter(db_train))[1].shape, next(iter(db_test))[0].shape, next(iter(db_test))[1].shape)