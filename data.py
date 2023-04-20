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

    x = np.array(x)
    y = np.array(y)
    print("x example:")
    print(x[:5,:])
    print("y example:")
    print(y[:5])

    db = tf.data.Dataset.from_tensor_slices((x,y))
    db = db.shuffle(94).batch(8)
    return db, len(v2i)

if __name__ == "__main__":
    db, vocab_num = load_w2v(method="cbow")
    print(next(iter(db))[0].shape, next(iter(db))[1].shape, vocab_num)