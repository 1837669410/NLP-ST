import tensorflow as tf
import time
import numpy as np
from tensorflow import keras
from utils import set_soft_gpu
from data import load_mrpc

class ELMo(keras.Model):
    def __init__(self, vocab_num, emb_dim, units, n_layer, use_ln=False):
        super().__init__()
        self.n_layer = n_layer
        self.units = units
        self.use_ln = use_ln

        self.embedding = keras.layers.Embedding(input_dim=vocab_num, output_dim=emb_dim,
                                                embeddings_initializer=keras.initializers.RandomNormal(0., 0.001),
                                                mask_zero=True)
        # forward lstm
        self.fs = [keras.layers.LSTM(units=units, return_sequences=True, return_state=True) for _ in range(n_layer)]
        if use_ln:
            self.fln = [keras.layers.LayerNormalization() for _ in range(n_layer)]
        self.f_dense = keras.layers.Dense(vocab_num)
        # backward lstm
        self.bs = [keras.layers.LSTM(units=units, return_sequences=True, return_state=True, go_backwards=True) for _ in range(n_layer)]
        if use_ln:
            self.bln = [keras.layers.LayerNormalization() for _ in range(n_layer)]
        self.b_dense = keras.layers.Dense(vocab_num)

        # loss_func
        self.f_loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.b_loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # opt
        self.opt = keras.optimizers.Adam(0.001)

    def call(self, x):
        # forward 0 1 2 3
        # predict 1 2 3 4
        # backward 4 3 2 1
        # predict 3 2 1 0
        emb = self.embedding(x)   # [None step] -> [None step emb_dim]
        mask = self.embedding.compute_mask(x)
        fxs, bxs = emb[:, :-1], emb[:, 1:]   # All positive sequences
        f_state, b_state = self.fs[0].get_initial_state(fxs), self.bs[0].get_initial_state(fxs)
        i = 0
        for f, b in zip(self.fs, self.bs):
            fxs, fh, fc = f(fxs, mask=mask[:, :-1], initial_state=f_state)
            if self.use_ln:
                fxs = self.fln[i](fxs)
            f_state = [fh, fc]
            bxs, bh, bc = b(bxs, mask=mask[:, 1:], initial_state=b_state)   # Returns a reverse sequence
            if self.use_ln:
                bxs = self.fln[i](bxs)
            bxs = tf.reverse(bxs, axis=[1])   # Because the input is a forward sequence, it needs to be reversed
            b_state = [bh, bc]
            i += 1
        return fxs, bxs

    def train(self, x):
        fxs, bxs = self.call(x)   # [None step] -> [None step-1 emb_dim]
        fo, bo = self.f_dense(fxs), self.b_dense(bxs)   # [None step-1 emb_dim] -> [None step-1 vocab_num]
        f_loss, b_loss = self.f_loss_func(x[:, 1:], fo), self.b_loss_func(x[:, :-1], bo)
        loss = (f_loss + b_loss) / 2
        return fo, bo, loss

def train():
    set_soft_gpu(True)
    epoch = 100
    emb_dim = 256
    units = 256
    n_layer = 2
    use_save = False
    use_pretrain_model = False
    use_ln = False   # Using LayerNormalization to accelerate training can actually increase the speed by nearly 4-5 times
    s1, s2, i2v, v2i = load_mrpc(num_train_sample=4000, num_test_sample=500, min_freq=0, max_length=30, batch_size=64)
    model = ELMo(vocab_num=len(i2v), emb_dim=emb_dim, units=units, n_layer=n_layer, use_ln=use_ln)
    start_time = time.time()
    if use_pretrain_model:
        _, _, _ = model.train(next(iter(s1)))
        print("Loading pre trained model ......")
        model.embedding.set_weights(np.load("./model/elmo/elmo_embedding.npy"))
        model.fs[0].set_weights(np.load("./model/elmo/elmo_fs0.npy", allow_pickle=True))
        model.fs[1].set_weights(np.load("./model/elmo/elmo_fs1.npy", allow_pickle=True))
        model.f_dense.set_weights(np.load("./model/elmo/elmo_fdense.npy", allow_pickle=True))
        model.bs[0].set_weights(np.load("./model/elmo/elmo_bs0.npy", allow_pickle=True))
        model.bs[1].set_weights(np.load("./model/elmo/elmo_bs1.npy", allow_pickle=True))
        model.b_dense.set_weights(np.load("./model/elmo/elmo_bdense.npy", allow_pickle=True))
    for e in range(epoch):
        for step, x in enumerate(s1):
            with tf.GradientTape() as tape:
                fo, bo, loss = model.train(x)
                grads = tape.gradient(loss, model.trainable_variables)
            model.opt.apply_gradients(zip(grads, model.trainable_variables))

            if step % 40 == 0:
                f_pred = fo[0].numpy().argmax(axis=1)
                b_pred = bo[0].numpy().argmax(axis=1)
                print("epoch:%d | step:%d | time:%.2f | loss:%.3f"%(e, step, time.time()-start_time, loss))
                print("y_true: {}".format(" ".join([i2v[i] for i in x[0].numpy() if i2v[i] != "<pad>"])))
                print("f_true: {}".format(" ".join(["    "] + [i2v[i] for i in f_pred if i2v[i] != "<pad>"])))
                print("b_true: {}".format(" ".join([i2v[i] for i in b_pred if i2v[i] != "<pad>"])))

    if use_save:
        print("Saving model weights ......")
        np.save("./model/elmo/elmo_embedding.npy", model.embedding.weights)
        np.save("./model/elmo/elmo_fs0.npy", model.fs[0].weights)
        np.save("./model/elmo/elmo_fs1.npy", model.fs[1].weights)
        np.save("./model/elmo/elmo_fdense.npy", model.f_dense.weights)
        np.save("./model/elmo/elmo_bs0.npy", model.bs[0].weights)
        np.save("./model/elmo/elmo_bs1.npy", model.bs[1].weights)
        np.save("./model/elmo/elmo_bdense.npy", model.b_dense.weights)

if __name__ == "__main__":
    train()

