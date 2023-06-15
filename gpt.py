import tensorflow as tf
import time
import numpy as np
from tensorflow import keras
from data import load_mrpc_gpt
from utils import set_soft_gpu

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, model_dim, n_head, dropout_rate, use_bias=False):
        super().__init__()
        self.model_dim = model_dim
        self.n_head = n_head
        self.head_dim = model_dim // n_head

        self.wq = keras.layers.Dense(n_head * self.head_dim, use_bias=use_bias)
        self.wk = keras.layers.Dense(n_head * self.head_dim, use_bias=use_bias)
        self.wv = keras.layers.Dense(n_head * self.head_dim, use_bias=use_bias)
        self.o_dense = keras.layers.Dense(model_dim)
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, q, k, v, mask, training):
        _q = self.wq(q)   # [None q_step model_dim] -> [None q_step n_head*head_dim]
        _k, _v = self.wk(k), self.wv(v)   # [None q_step model_dim] -> [None q_step n_head*head_dim]
        _q = self.split_head(_q)   # [None q_step n_head*head_dim] -> [None n_head step head_dim]
        _k, _v= self.split_head(_k), self.split_head(_v)   # [None q_step n_head*head_dim] -> [None n_head step head_dim]
        context = self.scale_dot_product_attention(_q, _k, _v, mask)   # -> [None q_step n_head*head_dim]
        o = self.o_dense(context)   # [None q_step n_head*head_dim] -> [None q_step model_dim]
        o = self.dropout(o)
        return o

    def split_head(self, x):
        # input: [None step n_head*head_dim]
        # [None step n_head*head_dim] -> [None step n_head head_dim]
        x = tf.reshape(x, shape=[x.shape[0], x.shape[1], self.n_head, self.head_dim])
        # [None step n_head head_dim] -> [None n_head step head_dim]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def scale_dot_product_attention(self, q, k, v, mask=None):
        dk = tf.cast(k.shape[-1], dtype=tf.float32)
        # [None n_head q_step head_dim] @ [None n_head step head_dim] -> [None n_head q_step step]
        score = tf.matmul(q, k, transpose_b=True) / (tf.sqrt(dk) + 1e-9)
        if mask is not None:
            score += mask * -1e9
        attention = tf.nn.softmax(score, axis=-1)
        # [None n_head q_step step] @ [None n_head step head_dim] -> [None n_head q_step head_dim]
        context = tf.matmul(attention, v)
        # [None n_head q_step head_dim] -> [None q_step n_head head_dim]
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        # [None q_step n_head head_dim] -> [None q_step n_head*head_dim]
        context = tf.reshape(context, shape=[context.shape[0], context.shape[1], -1])
        return context

class PositionWiseFFN(keras.layers.Layer):
    def __init__(self, model_dim):
        super().__init__()
        self.o_dense1 = keras.layers.Dense(model_dim*4, activation="relu")
        self.o_dense2 = keras.layers.Dense(model_dim)

    def call(self, x):
        o = self.o_dense1(x)
        o = self.o_dense2(o)
        return o

class EncoderLayer(keras.layers.Layer):
    def __init__(self, model_dim, n_head, dropout_rate, use_bias=False):
        super().__init__()
        self.ln = [keras.layers.LayerNormalization() for _ in range(2)]
        self.mha = MultiHeadAttention(model_dim, n_head, dropout_rate, use_bias)
        self.ffn = PositionWiseFFN(model_dim)
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, xz, mask, training):
        attention = self.mha.call(xz, xz, xz, mask, training)
        o1 = self.ln[0](self.dropout(attention, training) + xz)
        ffn = self.ffn.call(o1)
        o2 = self.ln[1](self.dropout(ffn, training) + o1)
        return o2

class Encoder(keras.layers.Layer):
    def __init__(self, model_dim, n_head, n_layer, dropout_rate, use_bias=False):
        super().__init__()
        self.l = [EncoderLayer(model_dim, n_head, dropout_rate, use_bias) for _ in range(n_layer)]

    def call(self, xz, mask, training):
        for l in self.l:
            xz = l.call(xz, mask, training)
        return xz

class GPT(keras.Model):
    def __init__(self, model_dim, n_head, n_layer, dropout_rate, max_length, n_vocab, max_seg=3, padding_idx=0, lr=1e-4):
        super().__init__()
        self.padding_idx = padding_idx
        self.n_vocab = n_vocab
        self.max_length = max_length

        self.s1s2embedding = keras.layers.Embedding(input_dim=n_vocab, output_dim=model_dim)
        self.seg_embedding = keras.layers.Embedding(input_dim=max_seg, output_dim=model_dim)
        self.position_emb = self.add_weight(
            name="pos_emb", shape=[1, max_length, model_dim], dtype=tf.float32,
            initializer=keras.initializers.RandomNormal(0., 0.01)
        )
        self.ln = keras.layers.LayerNormalization()
        self.encoder = Encoder(model_dim, n_head, n_layer, dropout_rate)
        self.mlm = keras.layers.Dense(n_vocab)
        self.nsp = keras.layers.Dense(2)

        self.loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        self.opt = keras.optimizers.Adam(lr)

    def call(self, seq, seg, training=False):
        # input ([None step] [None step])
        # [None step] -> [None step model_dim]
        emb = self.add_emb(seq, seg)
        emb = self.ln(emb)
        # [None step model_dim] -> [None step model_dim]
        z = self.encoder.call(emb, mask=self.mask(seq), training=training)
        # [None step model_dim] -> [None step n_vocab]
        mlm_logits = self.mlm(z)
        # [None step*model_dim] -> [None 2]
        nsp_logits = self.nsp(tf.reshape(z, shape=[z.shape[0], -1]))
        return mlm_logits, nsp_logits

    def train(self, seq, seg, p_seq, label, training=True):
        # train: seq[:, :-1] seg[:, :-1] predict: p_seg[:, 1:]
        mlm_logits, nsp_logits = self.call(seq, seg, training)
        pad_seq = tf.math.not_equal(p_seq, self.padding_idx)
        pred_loss = tf.reduce_mean(tf.boolean_mask(self.loss_func(p_seq, mlm_logits), pad_seq))
        nsp_loss = tf.reduce_mean(self.loss_func(label, nsp_logits))
        loss = pred_loss + 0.2 * nsp_loss
        return loss, mlm_logits

    def add_emb(self, seq, seg):
        emb = self.s1s2embedding(seq) + self.seg_embedding(seg) + self.position_emb
        return emb

    def mask(self, seq):
        mask = 1 - tf.linalg.band_part(tf.ones((self.max_length, self.max_length)), -1, 0)
        return mask

def train():
    set_soft_gpu(True)
    epoch = 100
    data, i2v, v2i, max_length = load_mrpc_gpt(min_freq=2)
    model = GPT(model_dim=256, n_head=4, n_layer=4, dropout_rate=0.2, max_length=max_length - 1, n_vocab=len(i2v), lr=1e-4)
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