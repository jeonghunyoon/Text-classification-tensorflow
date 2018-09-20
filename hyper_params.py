import tensorflow as tf


class HyperParams:
    @staticmethod
    def get_hyper_params():
        return tf.contrib.training.HParams(
            embedding_dim=256
        )