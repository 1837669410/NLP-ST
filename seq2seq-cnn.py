import tensorflow as tf
from tensorflow import keras
from utils import set_soft_gpu
from data import load_fra
from seq2seq import pad_data, Mask_CategoricalCrossentropy

class Seq2Seq(keras.Model):

    def __init__(self, en_vocab_num, fr_vocab_num, emb_dim, units, max_length):
        super(Seq2Seq, self).__init__()

        # encode by conv and pool
        self.en_embedding = keras.layers.Embedding(input_dim=en_vocab_num, output_dim=emb_dim)
        self.conv = [keras.layers.Conv2D(filters=units, kernel_size=[i,emb_dim], strides=1, padding="valid", activation="relu") for i in [2,3,4]]
        self.pool = [keras.layers.MaxPool2D(pool_size=[i,1], strides=1, padding="valid") for i in [max_length-1,max_length-2,max_length-3]]
        self.en_dense = keras.layers.Dense(units, activation="relu")
        self.dropout = keras.layers.Dropout(0.5)

        # decode
        self.de_embedding = keras.layers.Embedding(input_dim=fr_vocab_num, output_dim=emb_dim)
        self.de_lstm = keras.layers.LSTM(units=units, return_state=True, return_sequences=True, dropout=0.1)
        self.de_dense = keras.layers.Dense(fr_vocab_num)

        # loss_func
        self.mask_loss_func = Mask_CategoricalCrossentropy()
        # opt
        self.opt = keras.optimizers.Adam(0.01)

    def Encode(self, en):
        # embedding [None max_length] -> [None max_length emb_dim] -> [None max_length emb_dim 1]
        emb = self.en_embedding(en)
        emb = tf.expand_dims(emb, axis=3)
        # conv [None max_length emb_dim 1] -> [None max_length-2+1 1 units] [None max_length-3+1 1 units] [None max_length-4+1 1 units]
        out1, out2, out3 = self.dropout(self.conv[0](emb)), self.dropout(self.conv[1](emb)), self.dropout(self.conv[2](emb))
        # pool [None max_length-2+1 1 units] [None max_length-3+1 1 units] [None max_length-4+1 1 units] -> [None 1 1 units]
        out1, out2, out3 = self.pool[0](out1), self.pool[1](out2), self.pool[2](out3)
        # concat 3*[None 1 1 units] -> [None 1 1 3*units]
        out = tf.concat((out1, out2, out3), axis=3)
        # squeeze [None 1 1 3*units] -> [None 3*units]
        out = tf.squeeze(out, axis=[1,2])
        # dense [None 3*units] -> [None units]
        h = self.en_dense(out)
        return h, h

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
        h, c = self.Encode(en)
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
        h, c = self.Encode(en)   # [1 max_length units] [1 units] [1 units]
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
    model = Seq2Seq(en_vocab_num=len(en_i2v), fr_vocab_num=len(fr_i2v), emb_dim=32, units=32, max_length=max_length)
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