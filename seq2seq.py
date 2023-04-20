import tensorflow as tf
from tensorflow import keras
from utils import set_soft_gpu
from data import load_fra

class Seq2Seq(keras.Model):

    def __init__(self, en_vocab_num, fr_vocab_num, emb_dim, units):
        super(Seq2Seq, self).__init__()

        # encode
        self.en_embedding = keras.layers.Embedding(input_dim=en_vocab_num, output_dim=emb_dim)
        self.en_lstm = keras.layers.LSTM(units=units, return_sequences=True, return_state=True)

        # decode
        self.de_embedding = keras.layers.Embedding(input_dim=fr_vocab_num, output_dim=emb_dim)
        self.de_lstm = keras.layers.LSTM(units=units, return_sequences=True, return_state=True)
        self.de_dense = keras.layers.Dense(fr_vocab_num)

    def Encode(self, en):
        # embedding [None max_length] -> [None max_length emb_dim]
        emb = self.en_embedding(en)
        # lstm [None max_length emb_dim] -> [None units]
        o, h, c = self.en_lstm(emb)
        return o, h, c

    def Decode(self, fr, state):
        # state [[None units] [None units]]
        # embedding [None max_length] -> [None max_length emb_dim]
        emb = self.de_embedding(fr)
        # context [None units] -> [units 1 units] -> [None max_length units]   get final state
        context = tf.repeat(tf.expand_dims(state[1], axis=1), emb.shape[1], axis=1)
        # concat [None max_length emb_dim] and [None max_length units] -> [None max_length emb_dim+units]   final state and emb
        emb_and_context = tf.concat((emb, context), axis=2)
        # lstm [None max_length emb_dim+units] -> [None max_length units]   init de_lstm state by en_lstm[h,c]
        o, h, c = self.de_lstm(emb_and_context, initial_state=state)
        return o, h, c

def train():
    set_soft_gpu(True)
    db_en, db_fr, en_i2v, en_v2i, fr_i2v, fr_v2i = load_fra(num_sample=600, min_freq=2, max_length=8)
    print(next(iter(db_en))[0].shape, next(iter(db_en))[1].shape, next(iter(db_fr))[0].shape, next(iter(db_fr))[1].shape)
    model = Seq2Seq(en_vocab_num=len(en_i2v), fr_vocab_num=len(fr_i2v), emb_dim=32, units=32)
    o, h, c = model.Encode(next(iter(db_en))[0])
    o, h, c = model.Decode(next(iter(db_fr))[0], [h,c])

if __name__ == "__main__":
    train()