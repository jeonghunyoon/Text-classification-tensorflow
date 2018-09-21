import tensorflow as tf


class HyperParams:
    @staticmethod
    def get_cnn_hyper_params():
        return tf.contrib.training.HParams(
            embedding_dim=256,
            filters=512,
            filter_sizes=[3, 4, 5],
            strides=(1, 1),
            learning_rate=0.001,
            beta2=0.99,
            drop_prob=0.5,
            dense_layer_units=1024,
            num_epochs=30,
            batch_size=128,
        )