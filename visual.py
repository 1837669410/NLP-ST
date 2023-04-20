import matplotlib.pyplot as plt
import numpy as np

def show_tfidf(tfidf, vocab):
    plt.imshow(tfidf, cmap="winter_r")
    plt.xticks(ticks=np.arange(tfidf.shape[1]), labels=vocab, fontsize=7, rotation=90)
    plt.yticks(ticks=np.arange(tfidf.shape[0]), fontsize=7)
    plt.title("tfidf", fontsize=12)
    plt.show()

def show_w2v(emb_weight, i2v, name):
    plt.figure()
    for i, v in i2v.items():
        if v not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            plt.text(emb_weight[i,0], emb_weight[i,1], s=v, c="red")   # word -> red
        else:
            plt.text(emb_weight[i,0], emb_weight[i,1], s=v, c="blue")   # num -> blue
    plt.xlim(np.min(emb_weight[:,0])-0.5, np.max(emb_weight[:,0]+0.5))
    plt.ylim(np.min(emb_weight[:,1])-0.5, np.max(emb_weight[:,1]+0.5))
    plt.xlabel("emb_x")
    plt.ylabel("emb_y")
    plt.title(name)
    plt.savefig("./visual/{}{}".format(name, ".jpg"))
    plt.show()