from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from visual import show_tfidf
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

def get_query_doc(q_score, n=3):
    q_index = np.argsort(q_score)[::-1][:n]   # get query result index
    result = []
    for i in q_index:
        result.append(docs[i])
    return result

model = TfidfVectorizer()
tfidf = model.fit_transform(docs)
print("idf:{}".format([(v,idf) for v, idf in zip(model.get_feature_names_out(), model.idf_)]))
print("vocab:{}".format(model.get_feature_names_out()))

q = "I get a coffee cup"
q_tfidf = model.transform([q])
score = cosine_similarity(q_tfidf, tfidf)
result = get_query_doc(score.squeeze())
print("query result:{}".format(result))
# show tfidf picture
show_tfidf(tfidf.todense(), model.get_feature_names_out())