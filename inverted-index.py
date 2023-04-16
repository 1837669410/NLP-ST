# inverted-index
import itertools

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

# 1、get words
words = [v.replace(",", "").split(" ") for v in docs]
words = list(set(itertools.chain(*words)))
# 2、create vocab to doc index
v2i = {}
for w in words:
    temp = []
    for i in range(len(docs)):
        if w in docs[i]:
            temp.append(i)
    v2i[w] = temp
# 3、get doc by keyword
key_word = "bob"
print("keyword:{}".format(key_word))
for i in v2i[key_word]:
    print("doc_id:{} | doc_content:{}".format(i, docs[i]))