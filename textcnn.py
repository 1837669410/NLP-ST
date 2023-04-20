import tensorflow as tf
from tensorflow import keras
from utils import set_soft_gpu
from data import load_imdb

class TextCNN(keras.Model):

    def __init__(self, vocab_num, emb_dim):
        super(TextCNN, self).__init__()

        # embedding
        self.embedding = keras.layers.Embedding(input_dim=vocab_num, output_dim=emb_dim,
                                                embeddings_initializer=keras.initializers.RandomNormal())
        # conv
        self.conv1 = [keras.layers.Conv2D(filters=100, kernel_size=[i,emb_dim], strides=1, padding="valid", activation="relu", activity_regularizer="l2") for i in [2,3,4]]
        self.conv2 = [keras.layers.Conv2D(filters=100, kernel_size=[i,emb_dim], strides=1, padding="valid", activation="relu", activity_regularizer="l2") for i in [2,3,4]]
        # pool
        self.pool = [keras.layers.MaxPool2D(pool_size=[i,1], strides=1, padding="valid") for i in [99,98,97]]
        # dense
        self.dense = keras.Sequential([
            keras.layers.Dense(100, activation="relu", activity_regularizer="l2"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(2)
        ])

        # loss_func
        self.loss_func = keras.losses.CategoricalCrossentropy(from_logits=True)
        # opt
        self.opt = keras.optimizers.Adam(0.0001, decay=0.001)

    def call(self, inputs, training=None, mask=None):
        # embedding [None max_len] -> [None max_len emb_dim] -> [None max_len emb_dim 1]
        emb = self.embedding(inputs)
        out = tf.expand_dims(emb, axis=3)
        # conv [None max_len emb_dim 1] -> [None max_len-2+1 1 100] [None max_len-3+1 1 100] [None max_len-4+1 1 100]
        out11, out12, out13 = self.conv1[0](out), self.conv1[1](out), self.conv1[2](out)
        out21, out22, out23 = self.conv2[0](out), self.conv2[1](out), self.conv2[2](out)
        # pool [None max_len-2+1 1 100] [None max_len-3+1 1 100] [None max_len-4+1 1 100] -> [None 1 1 100]
        out11, out12, out13 = self.pool[0](out11), self.pool[1](out12), self.pool[2](out13)
        out21, out22, out23 = self.pool[0](out21), self.pool[1](out22), self.pool[2](out23)
        # concat 6*[None 1 1 100] -> [None 1 1 600]
        out = tf.concat((out11, out21, out12, out22, out13, out23), axis=3)
        # squeeze [None 1 1 600] -> [None 600]
        out = tf.squeeze(out)
        # dense [None 600] -> [None 100] -> [None 2]
        out = self.dense(out)
        return out

    def loss(self, x, y):
        out = self.call(x)
        y = tf.one_hot(y, depth=2)
        loss = self.loss_func(y, out)
        return loss

def train():
    set_soft_gpu(True)
    vocab_num = 10000
    max_len = 100
    batch_size = 64
    epoch = 7
    db_train, db_test = load_imdb(vocab_num=vocab_num, maxlen=max_len, batch_size=batch_size)
    model = TextCNN(vocab_num=vocab_num, emb_dim=64)
    for e in range(epoch):
        for step, (x, y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                loss = model.loss(x,y)
                grads = tape.gradient(loss, model.trainable_variables)
            model.opt.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print("epoch:{} | step:{} | loss:{}".format(e, step, loss))

        total_num = 0
        total_acc = 0
        for step, (x, y) in enumerate(db_test):
            total_num += x.shape[0]
            out = model.call(x, training=False)   # [None 2]
            prob = tf.nn.softmax(out, axis=1)   # [None 2]
            pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)   # [None,]
            pred = tf.cast(tf.equal(pred, y), dtype=tf.int32)   # [None,]
            total_acc += tf.reduce_sum(pred)
        print("epoch:{} | acc:{}".format(e, total_acc / total_num))

if __name__ == "__main__":
    train()