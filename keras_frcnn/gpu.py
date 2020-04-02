import tensorflow as tf


def setup_gpu(gpu_id):
    import os
    if gpu_id == 'cpu' or gpu_id == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # # testing to set memory limit when initialising run (parameter from function input)
    # gpu_memory_limit = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=memory_limit)
    # config = tf.compat.v1.ConfigProto(gpu_options=gpu_memory_limit)

    config = tf.compat.v1.ConfigProto()

    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("***", gpus)
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print("***", len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print("***", e)