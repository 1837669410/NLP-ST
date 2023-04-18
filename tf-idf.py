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

def cosine_similarity(x, y):
    norm_x = np.sqrt(np.sum(np.square(x), axis=0, keepdims=True))   # [1, 1]
    norm_y = np.sqrt(np.sum(np.square(y), axis=0, keepdims=True))   # [1, n_doc]
    score = x.T.dot(y)   # [1, n_doc]
    return score / (norm_x*norm_y)

def query(q, idf, tfidf):
    q_words = q.replace(",", "").split(" ")
    unk_word = 0
    for w in q_words:
        if w not in vocab:
            unk_word += 1
            vocab.append(w)
            v2i[w] = len(v2i)
            i2v[len(v2i)-1] = w
    if unk_word > 0:
        _idf = np.concatenate((idf, np.zeros(shape=[unk_word, 1])), axis=0)   # [n_vocab+unk_word, 1]
        _tf_idf = np.concatenate((tfidf, np.zeros(shape=[unk_word, len(docs)])), axis=0)   # [n_vocab+unk_word, n_doc]
    else:
        _idf = idf
        _tf_idf = tfidf
    q_tf = np.zeros(shape=[len(_idf), 1])   # [n_vocab+unk_word, 1]
    word, word_count = np.unique(q_words, return_counts=True)
    for i in range(len(word)):
        q_tf[v2i[word[i]], 0] = word_count[i]   # [n_vocab+unk_word, 1]
    q_tf_idf = q_tf * _idf   # [n_vocab+unk_word, 1] get query vector
    q_score = cosine_similarity(q_tf_idf, _tf_idf)   # [1, n_doc]
    return q_score.squeeze()

def get_query_doc(q_score, n=3):
    q_index = np.argsort(q_score)[::-1][:n]   # get query result index
    result = []
    for i in q_index:
        result.append(docs[i])
    return result

tf = get_tf()   # [n_vocab, n_doc]
idf = get_idf()   # [n_vocab, 1]
tfidf = tf * idf   # [n_vocab, n_doc]
# get keywords
get_keywords()
# query
q = "I get a coffee cup"
score = query(q, idf, tfidf)   # get similarity
result = get_query_doc(score)   # get query result
print("query result:{}".format(result))