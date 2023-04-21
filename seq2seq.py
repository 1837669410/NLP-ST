import tensorflow as tf
from tensorflow import keras
from utils import set_soft_gpu
from data import load_fra
from d2l import tensorflow as d2l   # cite:https://zh.d2l.ai/chapter_recurrent-modern/seq2seq.html#sec-seq2seq-training

def pad_data(data, v2i, max_length):
    if len(data) > max_length:
        data = data[:max_length]
    else:
        data = data + (max_length - len(data)) * ["<pad>"]
    for i in range(len(data)):
        data[i] = v2i.get(data[i])
    return data

def sequence_mask(sequence, valid_len, value=0):
    # get mask tensor
    # sequence [None max_len]
    max_len = sequence.shape[1]   # get max_len
    mask = tf.range(start=0, limit=max_len, dtype=tf.float32)[None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)   # get mask mat
    if len(sequence.shape) == 3:
        return tf.where(tf.expand_dims(mask, axis=-1), sequence, value)
    else:
        return tf.where(mask, sequence, value)

class Mask_CategoricalCrossentropy(keras.losses.Loss):

    def __init__(self):
        super(Mask_CategoricalCrossentropy, self).__init__()

    def call(self, y_true, y_pred, valid_len):
        # y_true [None, max_length]
        # y_pred [None, max_length, vocab_num]
        # valid_len [None, ]
        weights = tf.ones_like(y_true, dtype=tf.float32)   # get weight
        weights = sequence_mask(weights, valid_len=valid_len)  # get weight mask
        y_true_hot = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=y_pred.shape[-1])   # [None max_length] -> [None max_length vocab_num]
        UnWeights_loss = keras.losses.CategoricalCrossentropy(from_logits=True, reduction="none")(y_true_hot, y_pred)   # get loss [None max_length]
        Weights_loss = UnWeights_loss * weights   # get loss [None max_length]
        Weights_loss = tf.reduce_mean(Weights_loss, axis=1)   # get loss [None, ]
        return Weights_loss

class Seq2Seq(keras.Model):

    def __init__(self, en_vocab_num, fr_vocab_num, emb_dim, units):
        super(Seq2Seq, self).__init__()

        # encode
        self.en_embedding = keras.layers.Embedding(input_dim=en_vocab_num, output_dim=emb_dim)
        self.en_lstm = keras.layers.LSTM(units=units, return_state=True, return_sequences=True, dropout=0.1)

        # decode
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

    def Decode(self, fr, state):
        # state [[None units] [None units]]
        # embedding [None max_length] -> [None max_length emb_dim]
        emb = self.de_embedding(fr)
        # context [None units] -> [units 1 units] -> [None max_length units]   get final state
        context = tf.repeat(tf.expand_dims(state[-1], axis=1), repeats=emb.shape[1], axis=1)
        # concat [None max_length emb_dim] and [None max_length units] -> [None max_length emb_dim+units] final state and emb
        emb_and_context = tf.concat((emb, context), axis=2)
        # lstm [None max_length emb_dim+units] -> [None max_length units] [None units] [None units] init de_lstm state by en_lstm[h,c]
        o, h, c = self.de_lstm(emb_and_context, initial_state=state)
        # dense [None max_length units] -> [None max_length vocab_num]
        o = self.de_dense(o)
        return o, h, c

    def train(self, en, fr, valid_len, fr_v2i):
        # en [None max_length] | fr [None max_length] | valid_len [None, ]
        # go [None, 1]   add <go>
        go = tf.reshape(tf.constant([fr_v2i['<go>']] * fr.shape[0], dtype=tf.float32),shape=(-1, 1))
        # concat [None 1] [None max_length-1] -> [None 1+max_length-1]   Forced learning
        de_fr = tf.concat((go, fr[:,:-1]), axis=1)
        # encode [None max_length] -> [None max_length units] [None units] [None units]
        o, h, c = self.Encode(en)
        # decode [None 1+max_length] -> [None max_length vocab_num] [None units] [None units]
        o, h, c = self.Decode(de_fr, [h, c])
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
            de_out, h, c = self.Decode(de_input, [h, c])   # [1 1 vocab_num] [1,units] [1,units]
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