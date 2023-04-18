import matplotlib.pyplot as plt
import numpy as np


def show_tfidf(tfidf, vocab):
    plt.imshow(tfidf, cmap="winter_r")
    plt.xticks(ticks=np.arange(tfidf.shape[1]), labels=vocab, fontsize=7, rotation=90)
    plt.yticks(ticks=np.arange(tfidf.shape[0]), fontsize=7)
    plt.title("tfidf", fontsize=12)
    plt.show()