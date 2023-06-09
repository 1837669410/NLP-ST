import d2l.tensorflow
import tensorflow as tf

def set_soft_gpu(soft_gpu):
    # Code referencing MOFAN https://mofanpy.com/
    if soft_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

def fix_learning_rate(learning_rate, decay, epoch):
    return learning_rate * 1 / (1 + decay * epoch)

def grad_clipping(grads, theta):
    theta = tf.constant(theta, dtype=tf.float32)
    new_grads = []
    for grad in grads:
        if isinstance(grad, tf.IndexedSlices):
            new_grads.append(tf.convert_to_tensor(grad))
        else:
            new_grads.append(grad)
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2).numpy() for grad in new_grads)))
    norm = tf.cast(norm, dtype=tf.float32)
    if tf.greater(norm, theta):
        for i, grad in enumerate(new_grads):
            new_grads[i] = grad * theta / norm
    else:
        new_grads = new_grads
    return new_grads