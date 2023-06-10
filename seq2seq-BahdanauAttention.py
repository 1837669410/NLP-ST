import tensorflow as tf
from tensorflow import keras
from seq2seq import sequence_mask, Mask_CategoricalCrossentropy, pad_data
from utils import set_soft_gpu
from data import load_fra

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

# additive Attention
# wv.T @ tanh(wq @ q + wk @ k)
class AdditiveAttention(keras.layers.Layer):
    def __init__(self, key_size, query_size, num_hiddens, dropout):
        super().__init__()
        self.w_k = keras.layers.Dense(num_hiddens, use_bias=False)
        self.w_q = keras.layers.Dense(num_hiddens, use_bias=False)
        self.w_v = keras.layers.Dense(1, use_bias=False)
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, query, key, value, valid_len):
        # query [None 1 units]
        # key [None step units]
        # value [None step units]
        query, key = self.w_q(query), self.w_k(key) # [None 1 units] -> [None 1 units] [None step units] -> [None step units]
        features = tf.expand_dims(query, axis=2) + tf.expand_dims(key, axis=1) # [None 1 1 units] + [None 1 step units]
        features = tf.tanh(features)
        score = tf.squeeze(self.w_v(features), axis=-1)   # [None 1 step units] -> [None 1 step 1] > [None 1 step]
        score = masked_softmax(score, valid_len)
        return tf.matmul(self.dropout(score), value)   # [None 1 step] @ [None step units] -> [None 1 units]

class Seq2Seq(keras.Model):

    def __init__(self, en_vocab_num, fr_vocab_num, emb_dim, units):
        super(Seq2Seq, self).__init__()

        # encode
        self.en_embedding = keras.layers.Embedding(input_dim=en_vocab_num, output_dim=emb_dim)
        self.en_lstm = keras.layers.LSTM(units=units, return_state=True, return_sequences=True, dropout=0.1)

        # decode
        self.attention = AdditiveAttention(units, units, units, 0.1)
        self.de_embedding = keras.layers.Embedding(input_dim=fr_vocab_num, output_dim=emb_dim)
        self.de_lstm = keras.layers.LSTM(units=units, return_state=True, return_sequences=True, dropout=0.1)
        self.de_dense = keras.layers.Dense(fr_vocab_num)

        # loss_func
        self.mask_loss_func = Mask_CategoricalCrossentropy()
        # opt
        self.opt = keras.optimizers.Adam(0.01)

    def Encode(self, en):
        # embedding [None max_length] -> [None max_length emb_dim]
        emb = self.en_embedding(en)
        # lstm [None max_length emb_dim] -> [None max_length units] [None units] [None units]
        o, h, c = self.en_lstm(emb)
        return o, h, c

    def Decode(self, fr, en_o, state, en_valid_len):
        # state [[None units] [None units]]
        # embedding [None max_length] -> [None max_length emb_dim]
        emb = self.de_embedding(fr)
        # [None max_length emb_dim] -> [max_length None emb_dim]
        emb = tf.transpose(emb, perm=(1,0,2))
        outputs = []
        for x in emb:
            # query [None units] -> [None 1 units]
            query = tf.expand_dims(state[-1], axis=1)
            # context [None 1 units]
            context = self.attention(query, en_o, en_o, en_valid_len)
            # concat (None 1 units and None 1 emb_dim) -> [None 1 units+emb_dim]
            x = tf.concat((context, tf.expand_dims(x, axis=1)), axis=-1)
            o, h, c = self.de_lstm(x, initial_state=state)
            state = [h, c]
            outputs.append(o)
        # [None step units] -> [None step fr_vocab_num]
        o = self.de_dense(tf.concat(outputs, axis=1))
        return o, state[0], state[1]

    def train(self, en, fr, valid_len, fr_v2i):
        # en [None max_length] | fr [None max_length] | valid_len [None, ]
        # go [None, 1]   add <go>
        go = tf.reshape(tf.constant([fr_v2i['<go>']] * fr.shape[0], dtype=tf.float32),shape=(-1, 1))
        # concat [None 1] [None max_length-1] -> [None 1+max_length-1]   Forced learning
        de_fr = tf.concat((go, fr[:,:-1]), axis=1)
        # encode [None max_length] -> [None max_length units] [None units] [None units]
        o, h, c = self.Encode(en)
        # decode [None 1+max_length] -> [None max_length vocab_num] [None units] [None units]
        o, h, c = self.Decode(de_fr, o, [h, c], valid_len)
        # loss
        loss = self.mask_loss_func.call(fr, o, valid_len)
        return loss

    def inference(self, en, fr, en_i2v, en_v2i, fr_i2v, fr_v2i, max_length):
        en = en.lower().split(' ') + ["<eos>"]   # add <eos> end word
        en_valid_len = len(en)   # get en valid_len
        en = pad_data(en, en_v2i, max_length)   # pad en -> [max_length, ]
        # format en
        en = tf.expand_dims(en, axis=0)   # [max_length, ] -> [1, max_length]
        # encode
        o, h, c = self.Encode(en)   # [1 max_length units] [1 units] [1 units]
        # format de_input
        de_input = tf.expand_dims(tf.constant([fr_v2i["<go>"]]), axis=0)   # [1,] -> [1,1]
        result = []   # result
        for _ in range(max_length):
            de_out, h, c = self.Decode(de_input, o, [h, c], tf.cast([en_valid_len], dtype=tf.int32))   # [1 1 vocab_num] [1,units] [1,units]
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
    epoch = 300
    max_length = 10
    db_en, db_fr, en_i2v, en_v2i, fr_i2v, fr_v2i = load_fra(num_sample=600, min_freq=2, max_length=max_length, batch_size=64)
    print(next(iter(db_en))[0].shape, next(iter(db_en))[1].shape, next(iter(db_fr))[0].shape, next(iter(db_fr))[1].shape)
    model = Seq2Seq(en_vocab_num=len(en_i2v), fr_vocab_num=len(fr_i2v), emb_dim=32, units=32)
    for e in range(epoch):
        step = 0

        # train
        for (en, en_valid), (fr, fr_valid) in zip(db_en, db_fr):
            with tf.GradientTape() as tape:
                step += 1
                loss = model.train(en, fr, fr_valid, fr_v2i)
                grads = tape.gradient(loss, model.trainable_variables)
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