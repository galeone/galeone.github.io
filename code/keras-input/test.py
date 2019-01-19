import os
import sys
import numpy as np
import tensorflow as tf


def get_positive(batch_size):
    train_images = np.zeros((50, 28, 28, 1), dtype=np.float32) + 1.
    train_labels = np.int8(np.zeros((50, 1)) + 1)
    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset.make_one_shot_iterator().get_next()


def get_negative(batch_size):
    train_images = np.zeros((50, 28, 28, 1), dtype=np.float32)
    train_labels = np.int8(np.zeros((50, 1)))

    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset.make_one_shot_iterator().get_next()


def tf_model(feature, is_training):
    with tf.variable_scope("tf_model", reuse=tf.AUTO_REUSE):
        net = tf.layers.Conv2D(
            8, (3, 3), strides=(2, 2),
            activation=tf.nn.relu)(feature)  # 13x13x8
        net = tf.layers.BatchNormalization()(net, is_training)
        net = tf.layers.Conv2D(
            1, (3, 3), strides=(2, 2), activation=tf.nn.relu)(net)  # 6x6x1
        net = tf.layers.Flatten()(net)  # 36
        net = tf.layers.Dense(1)(net)
        return net


def keras_model(input_type):
    # Without input layer it works in CASE 1 and CASE 2
    # With `input_shape=(28,28,1)` it fails in CASE 1 and CASE 2
    # With `tf.keras.layers.InputLayer((28, 28, 1))` fails in CASE 1,2
    # Conclusion: defining static input shapes makes Keras define different placeholders
    # when sharing the variables, thus making it fails.
    if input_type == "shape":
        first = [
            tf.keras.layers.Conv2D(
                8, (3, 3),
                strides=(2, 2),
                activation=tf.nn.relu,
                input_shape=(28, 28, 1))
        ]
    elif input_type == "layer":
        first = [
            tf.keras.layers.InputLayer((28, 28, 1)),
            tf.keras.layers.Conv2D(
                8,
                (3, 3),
                strides=(2, 2),
                activation=tf.nn.relu,
            ),
        ]
    elif input_type == "dynamic":
        first = [
            tf.keras.layers.Conv2D(
                8,
                (3, 3),
                strides=(2, 2),
                activation=tf.nn.relu,
            )
        ]

    return tf.keras.Sequential([
        *first,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            1, (3, 3), strides=(2, 2), activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1),
    ])


def get_bn_vars(collection):
    moving_mean, moving_variance = None, None
    for var in collection:
        name = var.name.lower()
        if "variance" in name:
            moving_variance = var
        if "mean" in name:
            moving_mean = var

    if moving_mean is not None and moving_variance is not None:
        return moving_mean, moving_variance
    raise ValueError("Unable to find moving mean and variance")


def main_keras(case, input_type):
    print("CASE: ", case)
    print("INPUT TYPE: ", input_type)
    tf.keras.backend.set_learning_phase(True)
    model = keras_model(input_type)

    positive, positive_labels = get_positive(10)
    negative, negative_labels = get_negative(10)

    model_true = model(positive)
    loss = tf.losses.sigmoid_cross_entropy(positive_labels, model_true)

    if case == 2:
        model_false = model(negative)
        loss += tf.losses.sigmoid_cross_entropy(negative_labels, model_false)

    model.summary()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print("update_ops: ", update_ops)  # ! no update ops in the default graph

    # Use the update ops of the model itself
    print("model.updates: ", model.updates)

    #with tf.control_dependencies(model.updates):
    train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

    mean, variance = get_bn_vars(model.variables)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        while True:
            try:
                loss_value, _ = sess.run([loss, train_op])
                print("loss: ", loss_value)
            except tf.errors.OutOfRangeError:
                break
        print(sess.run([mean, variance]))


def main_layers(case):
    print("CASE: ", case)
    positive, positive_labels = get_positive(10)
    negative, negative_labels = get_negative(10)

    # tf layers
    model_true = tf_model(positive, True)

    loss = tf.losses.sigmoid_cross_entropy(positive_labels, model_true)
    if case == 2:
        model_false = tf_model(negative, True)
        loss += tf.losses.sigmoid_cross_entropy(negative_labels, model_false)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

    mean, variance = get_bn_vars(tf.global_variables())
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        while True:
            try:
                loss_value, _ = sess.run([loss, train_op])
                print("loss: ", loss_value)
            except tf.errors.OutOfRangeError:
                break
        print(sess.run([mean, variance]))


if __name__ == "__main__":
    case = int(sys.argv[2])
    if sys.argv[1] == "keras":
        input_type = sys.argv[3]
        sys.exit(main_keras(case, input_type))
    elif sys.argv[1] == "layers":
        sys.exit(main_layers(case))
    print("Please specify first argument in [keras,layers]")
