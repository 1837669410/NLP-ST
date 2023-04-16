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
words = list(set(itertools.chain(*words)))

def get_tf(method="log"):
    # [n_vocab, n_doc]
    _tf = np.zeros(shape=[len(words), len(docs)])
    for i in range(len(words)):
        for j in range(len(docs)):
            if words[i] in docs[j]:
                _tf[i][j] += 1
    if method == "log":
        weighted_tf = np.log(1+_tf)
        return weighted_tf

def get_idf(method="log"):
    # [n_vocab, 1]
    _idf = np.zeros(shape=[len(words), 1])
    for i in range(len(words)):
        for j in range(len(docs)):
            if words[i] in docs[j]:
                _idf[i,0] += 1
    if method == "log":
        weighted_idf = 1 + np.log(len(docs) / (_idf+1))
        return weighted_idf

def get_keywords(n=3):
    for i in range(tfidf.shape[1]):
        print("doc{}: keyword:{}".format(i, np.array(words)[np.argsort(tfidf[:, i])[::-1][:n]]))

tf = get_tf()   # [n_vocab, n_doc]
idf = get_idf()   # [n_vocab, 1]
tfidf = tf * idf   # [n_vocab, n_doc]
# get keywords
get_keywords()