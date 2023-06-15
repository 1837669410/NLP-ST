# This Bert's method comes from https://mofanpy.com/tutorials/machine-learning/nlp/bert
import tensorflow as tf
import time
import numpy as np
from gpt import GPT
from data import load_mrpc_gpt
from utils import set_soft_gpu

class BERT(GPT):
    def __init__(self, model_dim, n_head, n_layer, dropout_rate, max_length, n_vocab, max_seg=3, padding_idx=0, lr=1e-4):
        super().__init__(model_dim, n_head, n_layer, dropout_rate, max_length, n_vocab, max_seg=3, padding_idx=0, lr=1e-4)

    def mask(self, seq):
        mask =  tf.linalg.band_part(1 - tf.eye(self.max_length, dtype=tf.float32), 0, 1)
        return mask  # [step, step]

def train():
    set_soft_gpu(True)
    epoch = 100
    data, i2v, v2i, max_length = load_mrpc_gpt(min_freq=2)
    model = BERT(model_dim=256, n_head=4, n_layer=4, dropout_rate=0.2, max_length=max_length - 1, n_vocab=len(i2v), lr=1e-4)
    start_time = time.time()
    for e in range(epoch):
        for step, (seq, seg, seq_valid_len, label) in enumerate(data):
            with tf.GradientTape() as tape:
                loss, pred = model.train(seq[:, :-1], seg[:, :-1], seq[:, 1:], label)
                grads = tape.gradient(loss, model.trainable_variables)
            model.opt.apply_gradients(zip(grads, model.trainable_variables))

            if step % 40 == 0:
                print("epoch:%d | step:%d | time:%.2f | loss:%.3f"%(e, step, time.time()-start_time, loss))
                print("y_true: {}".format(" ".join([i2v[idx] for idx in seq[0].numpy()[:np.sum(seq_valid_len[0])+3]])))
                print("y_pred: {}".format(" ".join(["    "] + [i2v[idx] for idx in pred[0].numpy().argmax(axis=1)[:np.sum(seq_valid_len[0])+2]])))

if __name__ == "__main__":
    train()