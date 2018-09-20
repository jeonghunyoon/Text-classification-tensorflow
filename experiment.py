import shutil
import datetime
import consts
import logging

import tensorflow as tf
from input_fn import InputFunction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassifierExperiment:
    def __init__(self, train_X, test_X, train_y, test_y, hparmas, model_fn):
        self.train_X = train_X
        self.test_X = test_X
        self.train_y = train_y
        self.test_y = test_y
        self.model_fn = model_fn
        self.hparams = hparmas
        self.run_config = tf.estimator.RunConfig(
            log_step_count_steps=100,
            tf_random_seed=43,
            model_dir=consts.CNN_MODEL_DIR
        )
        self.input_fn = InputFunction(hparmas)

    def __serving_input_fn(self):
        """Serving input function : used in model prediction """
        receiver_tensor = {
            consts.KEY_OF_INPUT:
                tf.placeholder(tf.float32, [None, consts.MAX_SEQUENCE_LENGTH, self.hparams.embedding_dim])
        }
        features = {
            key: tensor
            for key, tensor in receiver_tensor.items()
        }
        return tf.estimator.export.ServingInputReceiver(
            features, receiver_tensor
        )

    def _get_train_spec(self):
        train_input_fn, train_hook = self.input_fn.get_input_fn(
            self.train_X,
            self.train_y,
            mode=tf.estimator.ModeKeys.TRAIN,
            num_epochs=self.hparams.num_epochs,
            batch_size=self.hparams.batch_size
        )
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn,
            max_steps=self.hparams.max_steps,
            hooks=[train_hook]
        )
        return train_spec

    def _get_eval_spec(self):
        eval_input_fn, eval_hook = self.input_fn.get_input_fn(
            self.test_X,
            self.test_y,
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=self.hparams.batch_size
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            exporters=[
                tf.estimator.LatestExporter(
                    name='predict',
                    serving_input_receiver_fn=self.__serving_input_fn,
                    exports_to_keep=1
                )
            ],
            hooks=[eval_hook],
            steps=None,
        )
        return eval_spec

    def _create_estimator(self):
        estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            config=self.run_config,
            params=self.hparams
        )

        return estimator

    def run_train_and_evaluate(self, is_resumed_training):
        """
        - Creating Estimator
        - Creating train, eval spec
        - Define train, eval
        """
        if not is_resumed_training:
            logger.info("Removing previous artifacts.")
            shutil.rmtree(consts.CNN_MODEL_DIR, ignore_errors=True)
        else:
            logger.info("Removing previous artifacts.")

        start_time = datetime.datetime.utcnow()
        logger.info("Experiment start time", start_at=start_time.strftime("%H:%M:%S"))

        estimator = self._create_estimator()

        train_spec = self._get_train_spec()
        eval_spec = self._get_eval_spec()

        tf.estimator.train_and_evaluate(
            estimator=estimator,
            train_spec=train_spec,
            eval_spec=eval_spec
        )

        end_time = datetime.datetime.utcnow()
        logger.info("Experiment end time", end_at=end_time.strftime("%H:%M:%S"))

    def evaluate(self):
        train_eval_input_fn, train_eval_hook = self.input_fn.get_input_fn(
            self.train_X,
            self.train_y,
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=len(self.train_X)
        )
        test_eval_input_fn, test_eval_hook = self.input_fn.get_input_fn(
            self.test_X,
            self.test_y,
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=len(self.train_y)
        )

        estimator = self._create_estimator()

        train_eval_result = estimator.evaluate(
            input_fn=train_eval_input_fn,
            steps=1,
            hooks=[train_eval_hook]
        )
        logger.info("Train evaluation result", train_eval_result=train_eval_result)

        test_eval_result = estimator.evaluate(
            input_fn=test_eval_input_fn,
            steps=1,
            hooks=[test_eval_hook]
        )
        logger.info("Test evaluation result", test_eval_result=test_eval_result)
