def set_soft_gpu(soft_gpu):
    # Code referencing MOFAN https://mofanpy.com/
    import tensorflow as tf
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