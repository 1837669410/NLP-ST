import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils import set_soft_gpu, grad_clipping
from data import load_fra
from seq2seq import Mask_CategoricalCrossentropy, pad_data, sequence_mask

def masked_softmax(x, valid_lens):
    if valid_lens is None:
        return tf.nn.softmax(x, axis=-1)
    else:
        shape = x.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])
        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)
        x = sequence_mask(tf.reshape(x, shape=(-1, shape[-1])), valid_lens, value=-1e6)
        return tf.nn.softmax(tf.reshape(x, shape=shape), axis=-1)

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, model_dim, n_head, dropout_rate, use_bias=False):
        super().__init__()
        self.model_dim = model_dim
        self.n_head = n_head
        self.head_dim = model_dim // n_head
        self.wq = keras.layers.Dense(n_head * self.head_dim, use_bias=use_bias)
        self.wk = keras.layers.Dense(n_head * self.head_dim, use_bias=use_bias)
        self.wv = keras.layers.Dense(n_head * self.head_dim, use_bias=use_bias)

        self.o_dense = keras.layers.Dense(model_dim, use_bias=use_bias)
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, q, k, v, valid_lens, **kwargs):
        # q [None q_step model_dim]
        # k [None step model_dim]
        # v [None step model_dim]
        # valid_lens [None, ]
        _q = self.wq(q)   # [None q_step model_dim] -> [None q_step n_head*head_dim]
        _k, _v = self.wk(k), self.wv(v)   # [None step model_dim] -> [None step n_head*head_dim]
        _q = self.split_head(_q)   # [None q_step n_head*head_dim] -> [None*n_head q_step head_dim]
        _k, _v = self.split_head(_k), self.split_head(_v)   # [None step n_head*head_dim] -> [None*n_head step head_dim]
        if valid_lens is not None:
            # [None, ] -> [n_head, None]
            valid_lens = tf.repeat(valid_lens, repeats=self.n_head, axis=0)
        attention = self.scaled_dot_product_attention(_q, _k, _v, valid_lens, **kwargs)   # [None*n_head q_step head_dim]
        # [None*n_head q_step head_dim] -> [None n_head q_step head_dim]
        out = tf.reshape(attention, shape=[-1, self.n_head, attention.shape[1], attention.shape[2]])
        # [None n_head q_step head_dim] -> [None q_step n_head head_dim]
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        # [None q_step n_head head_dim] -> [None q_step n_head*head_dim]
        out = tf.reshape(out, shape=[out.shape[0], out.shape[1], -1])
        # [None q_step n_head*head_dim] -> [None q_step model_dim]
        return self.o_dense(out)

    def scaled_dot_product_attention(self, q, k, v, valid_lens, **kwargs):
        # q [None*n_head q_step head_dim]
        # k [None*n_head step head_dim]
        # v [None*n_head step head_dim]
        dk = tf.cast(k.shape[-1], dtype=tf.float32)
        # [None*n_head q_step head_dim] @ [None*n_head step head_dim] -> [None*n_head q_step step]
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
        # Mask where is the position of the pad
        self.attention_weights = masked_softmax(scores, valid_lens)
        # [None*n_head q_step step] @ [None*n_head step head_dim] -> [None*n_head q_step head_dim]
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), v)


    def split_head(self, x):
        # x [None step n_head*head_dim]
        # 1、[None step n_head*head_dim] -> [None step n_head head_dim]
        x = tf.reshape(x, shape=[x.shape[0], x.shape[1], self.n_head, self.head_dim])
        # 2、[None step n_head head_dim] -> [None n_head step head_dim]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        # 3、[None n_head step head_dim] -> [None*n_head step head_dim]
        x = tf.reshape(x, shape=[-1, x.shape[2], x.shape[3]])
        return x

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, model_dim, dropout_rate, max_len=1000):
        super().__init__()
        self.dropout = keras.layers.Dropout(dropout_rate)
        # Create a sufficiently long pos
        self.pos = np.zeros((1, max_len, model_dim))
        # [max_len 1] / [model_dim/2] -> [max_len model_dim/2]
        self.pe = np.arange(max_len, dtype=np.float32).reshape(-1,1) / np.power(10000, np.arange(0, model_dim, 2, dtype=np.float32) / model_dim)
        self.pos[:,:,0::2] = np.sin(self.pe)
        self.pos[:,:,1::2] = np.cos(self.pe)

    def call(self, x, **kwargs):
        x = x + self.pos[:, :x.shape[1], :]
        return self.dropout(x, **kwargs)

class PositionWiseFFN(keras.layers.Layer):
    def __init__(self, model_dim, **kwargs):
        super().__init__()
        self.dense1 = keras.layers.Dense(model_dim*4)
        self.relu = keras.layers.ReLU()
        self.dense2 = keras.layers.Dense(model_dim)

    def call(self, x):
        return self.dense2(self.relu(self.dense1(x)))

class EncodeLayer(keras.layers.Layer):
    def __init__(self, model_dim, n_head, dropout_rate, bias=False):
        super().__init__()
        self.ln = [keras.layers.LayerNormalization() for _ in range(2)]
        self.mha = MultiHeadAttention(model_dim, n_head, dropout_rate, bias)
        self.ffn = PositionWiseFFN(model_dim)
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, x, valid_lens, **kwargs):
        attention = self.mha.call(x, x, x, valid_lens, **kwargs)
        out1 = self.ln[0](self.dropout(attention, **kwargs) + x)
        ffn = self.ffn.call(out1)
        out2 = self.ln[1](self.dropout(ffn, **kwargs) + out1)
        return out2

class Encoder(keras.layers.Layer):
    def __init__(self, vocab_size, model_dim, n_head, n_layer, dropout_rate, bias=False, **kwargs):
        super().__init__()
        self.model_dim = model_dim
        self.embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=model_dim)
        self.pos_encoding = PositionalEncoding(model_dim, dropout_rate)
        self.ls = [EncodeLayer(model_dim, n_head, dropout_rate, bias) for _ in range(n_layer)]

    def call(self, x, valid_lens, **kwargs):
        # Because the position encoding value is between -1 and 1,
        # Therefore, the embedding value is multiplied by the square root of the embedding dimension for scaling,
        # Then add it to the position code.
        x = self.pos_encoding.call(self.embedding(x) * tf.math.sqrt(tf.cast(self.model_dim, dtype=tf.float32)), **kwargs)
        for l in self.ls:
            x = l.call(x, valid_lens, **kwargs)
        return x

class DecodeLayer(keras.layers.Layer):
    def __init__(self, model_dim, n_head, dropout_rate, i, bias=False, **kwargs):
        super().__init__()
        self.i = i
        self.ln = [keras.layers.LayerNormalization() for _ in range(3)]
        self.mha = [MultiHeadAttention(model_dim, n_head, dropout_rate, bias) for _ in range(2)]
        self.ffn = PositionWiseFFN(model_dim)
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, x, state, **kwargs):
        en_outputs, en_valid_lens = state[0], state[1]
        # During the training phase, all word elements of the output sequence are processed at the same time,
        # Therefore, state[2] [self.i] is initialized to None.
        # In the prediction stage, the output sequence is decoded one by one through word elements,
        # Therefore, state[2] [self.i] contains the output representation until the i-th block of the current time step is decoded
        if state[2][self.i] is None:
            key_values = x
        else:
            key_values = tf.concat((state[2][self.i], x), axis=1)
        state[2][self.i] = key_values   # Transfer information from the previous layer to the next layer
        if kwargs["training"]:
            batch_size, step, _ = x.shape
            # dec_valid_lens: (batch_size,num_steps),
            # Each row is [1,2,...,num_steps]
            de_valid_lens = tf.repeat(tf.reshape(tf.range(1, step+1), shape=(-1, step)), repeats=batch_size, axis=0)
        else:
            de_valid_lens = None

        attention1 = self.mha[0].call(x, key_values, key_values, de_valid_lens, **kwargs)
        out1 = self.ln[0](self.dropout(attention1, **kwargs) + x)
        attention2 = self.mha[1].call(out1, en_outputs, en_outputs, en_valid_lens, **kwargs)
        out2 = self.ln[1](self.dropout(attention2, **kwargs) + out1)
        ffn = self.ffn.call(out2)
        out3 = self.ln[2](self.dropout(ffn, **kwargs) + out2)
        return out3, state

class Decoder(keras.layers.Layer):
    def __init__(self, vocab_size, model_dim, n_head, n_layer, dropout_rate, **kwargs):
        super().__init__()
        self.model_dim = model_dim
        self.n_layer = n_layer
        self.embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=model_dim)
        self.pos_encoding = PositionalEncoding(model_dim, dropout_rate)
        self.ls = [DecodeLayer(model_dim, n_head, dropout_rate, i) for i in range(n_layer)]
        self.dense = keras.layers.Dense(vocab_size)

    def init_state(self, en_outputs, en_valid_lens, *args):
        return [en_outputs, en_valid_lens, [None] * self.n_layer]

    def call(self, x, state, **kwargs):
        x = self.pos_encoding(self.embedding(x) * tf.math.sqrt(tf.cast(self.model_dim, dtype=tf.float32)), **kwargs)
        for l in self.ls:
            x, state = l.call(x, state, **kwargs)
        return self.dense(x), state

class Transformer(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        # loss_func
        self.mask_loss_func = Mask_CategoricalCrossentropy()
        # opt
        self.opt = keras.optimizers.Adam(0.005)

    def call(self, en_x, de_x, *args, **kwargs):
        en_outputs = self.encoder.call(en_x, *args, **kwargs)
        # Repeatedly assigning the value of 'None' to state [2] during training
        de_state = self.decoder.init_state(en_outputs, *args)
        return self.decoder(de_x, de_state, **kwargs)

    def inference(self, en, fr, en_i2v, en_v2i, fr_i2v, fr_v2i, max_length):
        en = en.lower().split(' ') + ["<eos>"]   # add <eos> end word
        en_valid_len = len(en)   # get en valid_len
        en = pad_data(en, en_v2i, max_length)   # pad en -> [max_length, ]
        # format en
        en = tf.expand_dims(en, axis=0)   # [max_length, ] -> [1, max_length]
        # encode
        o = self.encoder(en, en_valid_len, training=False)   # [1 max_length units] [1 units] [1 units]
        de_state = self.decoder.init_state(o, en_valid_len)   # Assign None only once during inference
        # format de_input
        de_input = tf.expand_dims(tf.constant([fr_v2i["<go>"]]), axis=0)   # [1,] -> [1,1]
        result = []   # result
        for _ in range(max_length):
            de_out, de_state = self.decoder(de_input, de_state, training=False)   # [1 1 vocab_num] [1,units] [1,units]
            de_input = tf.argmax(de_out, axis=2)   # get next input
            pred = tf.squeeze(de_input, axis=0)   # get predict
            if pred == fr_v2i["<eos>"]:   # set stop condition
                break
            result.append(pred.numpy()[0])   # save predict result
        # translate de_output index to value
        for i in range(len(result)):
            result[i] = fr_i2v.get(result[i])
        return " ".join(result)

def train():
    set_soft_gpu(True)
    epoch = 200
    model_dim = 32
    n_head = 4
    n_layer = 2
    dropout_rate = 0.1
    max_length = 10
    db_en, db_fr, en_i2v, en_v2i, fr_i2v, fr_v2i = load_fra(num_sample=600, min_freq=2, max_length=max_length, batch_size=64)
    print(next(iter(db_en))[0].shape, next(iter(db_en))[1].shape, next(iter(db_fr))[0].shape, next(iter(db_fr))[1].shape)

    TransformerEncode = Encoder(vocab_size=len(en_i2v), model_dim=model_dim, n_head=n_head, n_layer=n_layer,
                                dropout_rate=dropout_rate)
    TransformerDecode = Decoder(vocab_size=len(fr_i2v), model_dim=model_dim, n_head=n_head, n_layer=n_layer,
                                dropout_rate=dropout_rate)
    model = Transformer(TransformerEncode, TransformerDecode)

    for e in range(epoch):

        # train
        for (en, en_valid), (fr, fr_valid) in zip(db_en, db_fr):
            with tf.GradientTape() as tape:
                go = tf.reshape(tf.constant([fr_v2i["<go>"]] * fr.shape[0], dtype=tf.float32), shape=(-1,1))   # add <go>
                de_input = tf.concat((go, fr[:,:-1]), axis=1)   # Teacher forcing
                y_hat, _ = model.call(en, de_input, en_valid, training=True)
                loss = model.mask_loss_func.call(fr, y_hat, fr_valid)
                grads = tape.gradient(loss, model.trainable_variables)
                grads = grad_clipping(grads, 1)
            model.opt.apply_gradients(zip(grads, model.trainable_variables))

        # inference
        if e % 5 == 0:
            print("epoch:{}".format(e))
            x = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
            y = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
            result = []
            for e, f in zip(x, y):
                result.append(model.inference(e, f, en_i2v, en_v2i, fr_i2v, fr_v2i, max_length=max_length))
            print("en:{}".format(x))
            print("true_fr:{}".format(y))
            print("pred_fr:{}".format(result))

if __name__ == "__main__":
    train()