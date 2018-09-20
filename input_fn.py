import tensorflow as tf
import logging

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
        """Input function 을 정의합니다."""

        iterator_initializer_hook = IteratorInitializerHook()

        def input_fn():
            shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
            buffer_size = 2 * self.hparams.batch_size + 1

            logger.info("Input message dimension", texts=texts.shape)
            logger.info("Input label dimension", labels=labels.shape)

            # Numpy array 를 tf.dataset 에서 사용할 경우, 메모리 부족 문제가 발생할 수 있어서 placeholder 를 사용합니다.
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
                    feed_dict = {
                        texts_placeholder: texts,
                        labels_placeholder: labels
                    }
                )
            logger.info("Dataset iterator created", features=features)
            logger.info("Dataset iterator created", targets=targets)

            return {'reports': features}, targets  # Dictionary 형태로 output 을 넘겨줍니다.

        return input_fn, iterator_initializer_hook


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Session 이 생성된 이후에, iterator data 의 초기화를 hook 하는 class 입니다."""
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Session 이 생성된 후에 iterator 를 초기화하는 함수입니다."""
        self.iterator_initializer_func(session)