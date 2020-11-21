import sys

import tensorflow as tf


def main() -> int:
    at = tf.saved_model.load("at")
    batch_size = 32

    at.learn(
        tf.zeros((batch_size, 1, 3), dtype=tf.float32),
        tf.zeros((batch_size), dtype=tf.int32),
    )
    at.predict(tf.zeros((batch_size, 1, 3), dtype=tf.float32))

    return 0


if __name__ == "__main__":
    sys.exit(main())
