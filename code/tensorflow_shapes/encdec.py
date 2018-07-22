import tensorflow as tf
import numpy as np


def main():

    inputs_ = tf.placeholder(tf.float32, shape=(None, None, None, None))
    inputs = tf.cond(
        tf.equal(tf.shape(inputs_)[-1], 3), lambda: inputs_,
        lambda: tf.image.grayscale_to_rgb(inputs_))
    print(inputs)
    inputs.set_shape((None, None, None, 3))
    layer1 = tf.layers.conv2d(
        inputs,
        32,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation=tf.nn.relu,
        name="layer1")
    layer2 = tf.layers.conv2d(
        layer1,
        32,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation=tf.nn.relu,
        name="layer2")
    encode = tf.layers.conv2d(
        layer2, 10, kernel_size=(6, 6), strides=(1, 1), name="encode")
    print(encode)
    d_layer2 = tf.image.resize_nearest_neighbor(encode, tf.shape(layer2)[1:3])
    d_layer2 = tf.layers.conv2d(
        d_layer2,
        32,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation=tf.nn.relu,
        padding="SAME",
        name="d_layer2")

    d_layer1 = tf.image.resize_nearest_neighbor(d_layer2, tf.shape(layer1)[1:3])
    d_layer1 = tf.layers.conv2d(
        d_layer1,
        32,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation=tf.nn.relu,
        padding="SAME",
        name="d_layer1")

    decode = tf.image.resize_nearest_neighbor(d_layer1, tf.shape(inputs)[1:3])
    decode = tf.layers.conv2d(
        decode,
        inputs.shape[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=tf.nn.tanh,
        padding="SAME",
        name="decode")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o = sess.run(decode, feed_dict={inputs_: np.zeros((1, 28, 80, 1))})
        print(o.shape)

    return 0


if __name__ == "__main__":
    sys.exit(main())
