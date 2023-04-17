# tf-idf
import itertools
import numpy as np

docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup",
]

words = [v.replace(",", "").split(" ") for v in docs]
vocab = list(set(itertools.chain(*words)))
i2v = {i: v for i, v in enumerate(vocab)}   # index to value
v2i = {v: i for i, v in i2v.items()}   # value to index

def get_tf(method="log"):
    # [n_vocab, n_doc]
    _tf = np.zeros(shape=[len(vocab), len(docs)])
    for i in range(len(words)):
        word, word_count = np.unique(words[i], return_counts=True)
        for j in range(len(word)):
            _tf[v2i[word[j]], i] = word_count[j]
    if method == "log":
        return np.log(1+_tf)

def get_idf(method="log"):
    # [n_vocab, 1]
    _idf = np.zeros(shape=[len(vocab), 1])
    for i in range(len(vocab)):
        temp = 0
        for j in range(len(words)):
            if vocab[i] in words[j]:
                temp += 1
        _idf[i, 0] = temp
    if method == "log":
        return 1 + np.log(len(docs) / (_idf+1))

def get_keywords(n=3):
    for i in range(tfidf.shape[1]):
        print("doc{}: keyword:{}".format(i, np.array(vocab)[np.argsort(tfidf[:, i])[::-1][:n]]))

tf = get_tf()   # [n_vocab, n_doc]
idf = get_idf()   # [n_vocab, 1]
tfidf = tf * idf   # [n_vocab, n_doc]
# get keywords
get_keywords()