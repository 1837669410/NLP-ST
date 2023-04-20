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

def preprocess_imdb(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x, y

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
    db_train = db_train.map(preprocess_imdb).shuffle(25000).batch(batch_size)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.map(preprocess_imdb).shuffle(25000).batch(batch_size)
    return db_train, db_test

def no_space(v, pre_v):
    if v in [",", ".", "!", "?"] and pre_v != " ":
        return " " + v
    else:
        return v

def get_vocab(data, min_freq=2):
    data_vocab, data_vocab_count = np.unique(data, return_counts=True)   # get vocab and vocab_count
    data_vocab_index = np.where(data_vocab_count >= min_freq)   # get vocab_count >= 2 index
    data_vocab = data_vocab[data_vocab_index]   # get vocab
    return data_vocab

def get_i2v_v2i(vocab):
    i2v = {i+4: v for i, v in enumerate(vocab)}
    i2v[0] = "<unk>"   # stop word
    i2v[1] = "<pad>"   # padding word
    i2v[2] = "<go>"    # start word
    i2v[3] = "<eos>"   # end word
    v2i = {v: i for i, v in i2v.items()}
    return i2v, v2i

def build_nmt_datasets(data, i2v, v2i, max_length=8):
    valid_len = []
    for i in range(len(data)):
        data[i].append("<eos>")   # add stop word
        valid_len.append(len(data[i]))
        if len(data[i]) > max_length:
            data[i] = data[i][:max_length]   # truncate
        else:
            data[i] = data[i] + (max_length - len(data[i])) * ["<pad>"]   # pad
        for j in range(len(data[i])):
            data[i][j] = v2i.get(data[i][j], v2i.get("<unk>"))
    return np.array(data), np.array(valid_len)

def preprocess_nmt(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x, y

def load_fra(num_sample=600, min_freq=2, max_length=8):
    with open("./data/fra.txt", "r", encoding="utf-8") as fp:
        data = fp.read()
    data = data.replace('\u202f', ' ').replace('\xa0', ' ').lower()   # standard format
    data = "".join([no_space(data[i], data[i-1]) for i in range(len(data))])   # Make sure there are spaces before [,.!?]
    en = []   # english
    fr = []   # french
    for i, v in enumerate(data.split("\n")):
        if i >= num_sample:   # set max num_sample
            break
        temp = v.split("\t")
        en.append(temp[0].split(" "))   # get en
        fr.append(temp[1].split(" "))   # get fr
    en_vocab = get_vocab(list(itertools.chain(*en)), min_freq=min_freq)   # get en vocab
    fr_vocab = get_vocab(list(itertools.chain(*fr)), min_freq=min_freq)   # get fr vocab
    en_i2v, en_v2i = get_i2v_v2i(en_vocab)   # get en: index to vocab and vocab to index
    fr_i2v, fr_v2i = get_i2v_v2i(fr_vocab)   # get fr: index to vocab and vocab to index
    en, valid_en = build_nmt_datasets(en, en_i2v, en_v2i, max_length=max_length)   # get train en
    fr, valid_fr = build_nmt_datasets(fr, fr_i2v, fr_v2i, max_length=max_length)   # get train fr
    db_en = tf.data.Dataset.from_tensor_slices((en, valid_en))
    db_en = db_en.map(preprocess_nmt).shuffle(num_sample).batch(64)   # get db_en iter
    db_fr = tf.data.Dataset.from_tensor_slices((fr, valid_fr))
    db_fr = db_fr.map(preprocess_nmt).shuffle(num_sample).batch(64)   # get db_fr iter
    return db_en, db_fr, en_i2v, en_v2i, fr_i2v, fr_v2i

if __name__ == "__main__":
    load_fra(num_sample=600)