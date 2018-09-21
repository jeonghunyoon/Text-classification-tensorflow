import tensorflow as tf
import consts


def get_convolution_pooling_layer(input_layer, params, i):
    """
    - Convolution layer + max pooling layer
    - Kernel size : (height, width)
    - Pool size : (height, width)
    """

    # Convolution layers
    conv_layer = tf.layers.conv2d(inputs=input_layer,
                                  filters=params.filters,  # Filter 의 수
                                  kernel_size=(params.filter_sizes[i], params.embedding_dim),
                                  padding='valid',
                                  activation=tf.nn.relu)
    # Max pool layer
    pool_layer = tf.layers.max_pooling2d(inputs=conv_layer,
                                         pool_size=(consts.MAX_SEQUENCE_LENGTH - params.filter_sizes[i] + 1, 1),
                                         strides=params.strides)
    return pool_layer


def model_fn(features, labels, mode, params):
    """
    - Model function if tf.Estimator
    - features : return value of input_fn (dictionary form)
    - labels : return value of input_fn
    - Mode, params : defined at Estimator object
    """

    # Hyper parameters
    sequence_length = consts.MAX_SEQUENCE_LENGTH
    embedding_dim = params.embedding_dim
    learning_rate = params.learning_rate
    beta2 = params.beta2
    drop_prob = params.drop_prob
    units = params.dense_layer_units

    # Input shape : [batch size, height, width, channels]
    input_layer = tf.reshape(features[consts.KEY_OF_INPUT], shape=(-1, sequence_length, embedding_dim, 1))

    # Convolution & pooling layer
    layer_0 = get_convolution_pooling_layer(input_layer, params, 0)
    layer_1 = get_convolution_pooling_layer(input_layer, params, 1)
    layer_2 = get_convolution_pooling_layer(input_layer, params, 2)

    # Dense layer
    concatenated_tensor = tf.concat([layer_0, layer_1, layer_2], axis=1)
    flat_layer = tf.layers.flatten(concatenated_tensor)
    dense_layer = tf.layers.dense(inputs=flat_layer, units=units, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense_layer, rate=drop_prob, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer (units : num
    logits = tf.layers.dense(inputs=dropout, units=consts.NUM_LABELS)

    # EstimatorSpec for Prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)
        predictions = {
            'class': tf.gather(params.using_labels, predicted_indices),
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)  # signature_def_key of serving
        }
        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            export_outputs=export_outputs
        )

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    )

    # EstimatorSpec for Train
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta2=beta2)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op
        )

    # EstimatorSpec for Eval
    if mode == tf.estimator.ModeKeys.EVAL:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(tf.argmax(labels, 1), predicted_indices),
            'area_under_auc': tf.metrics.auc(labels, probabilities)
        }
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )
