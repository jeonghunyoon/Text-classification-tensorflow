import tensorflow as tf
import logging
import consts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFunction:
    """
    - Define input function of train spec and eval spec.
    - Use tf.data.Dataset
    - Because of memory, feeding input data into placeholders and use IteratorInitializerHook.
    """

    def __init__(self, hparmas):
        self.hparams = hparmas

    def get_input_fn(self, texts, labels,
                     mode=tf.estimator.ModeKeys.EVAL,
                     num_epochs=1,
                     batch_size=128):
        # For data feeding.
        iterator_initializer_hook = IteratorInitializerHook()

        def input_fn():
            shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
            buffer_size = 2 * self.hparams.batch_size + 1

            logger.info("Input message dimension", texts=texts.shape)
            logger.info("Input label dimension", labels=labels.shape)

            # In tf.dataset, numpy array => use placeholder. (Memory)
            texts_placeholder = tf.placeholder(texts.dtype, texts.shape)
            labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

            dataset = tf.data.Dataset.from_tensor_slices((texts_placeholder, labels_placeholder))

            if shuffle:
                dataset = dataset.shuffle(buffer_size)

            dataset = dataset.batch(self.hparams.batch_size)
            dataset = dataset.repeat(self.hparams.num_epochs)
            dataset = dataset.prefetch(buffer_size)

            iterator = dataset.make_initializable_iterator()
            features, targets = iterator.get_next()

            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={
                        texts_placeholder: texts,
                        labels_placeholder: labels
                    }
                )

            return {consts.KEY_OF_INPUT: features}, targets  # Dictionary type
        return input_fn, iterator_initializer_hook


class IteratorInitializerHook(tf.train.SessionRunHook):
    """After creating session, initiate iterator data hook."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        self.iterator_initializer_func(session)
