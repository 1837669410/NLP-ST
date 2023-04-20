import tensorflow as tf
from tensorflow import keras
from utils import set_soft_gpu
from data import load_w2v
from visual import show_w2v

class Skip_Gram(keras.Model):

    def __init__(self, vocab_num, emb_dim):
        super(Skip_Gram, self).__init__()
        self.vocab_num = vocab_num

        # embedding
        self.embedding = keras.layers.Embedding(input_dim=vocab_num, output_dim=emb_dim,
                                                embeddings_initializer=keras.initializers.RandomNormal(0., 0.1))

        # nce_w, nce_b
        self.nce_w = self.add_weight(
            name="nce_w",
            shape=[vocab_num, emb_dim],
            initializer=keras.initializers.TruncatedNormal(0., 0.1)
        )
        self.nce_b = self.add_weight(
            name="nce_b",
            shape=[vocab_num,],
            initializer=keras.initializers.Constant(0.1)
        )

        # opt
        self.opt = keras.optimizers.Adam(0.01)

    def call(self, inputs, training=None, mask=None):
        # embedding [None 1] -> [None 1 2] -> [None 2]
        emb = self.embedding(inputs)
        out = tf.squeeze(emb)
        return out

    def loss(self, x, y, training):
        loss = tf.nn.nce_loss(weights=self.nce_w, biases=self.nce_b, labels=tf.expand_dims(y, axis=1),
                              inputs=self.call(x, training), num_sampled=5, num_classes=self.vocab_num)
        loss = tf.reduce_mean(loss)
        return loss

def train():
    set_soft_gpu(True)  # set gpu mode
    db, vocab_num, i2v = load_w2v(method="skip_gram")
    model = Skip_Gram(vocab_num, 2)
    epoch = 100
    for e in range(epoch):
        for step, (x, y) in enumerate(db):
            with tf.GradientTape() as tape:
                loss = model.loss(x, y, True)
                grads = tape.gradient(loss, model.trainable_variables)
            model.opt.apply_gradients(zip(grads, model.trainable_variables))

            if step % 2 == 0:
                print("epoch:{} | step:{} | loss:{}".format(e, step, loss))

    show_w2v(model.get_weights()[0], i2v, "skip_gram")


if __name__ == "__main__":
    train()