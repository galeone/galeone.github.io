import sys

import tensorflow as tf
import tensorflow.keras as k


class ActivityTracker(tf.Module):
    def __init__(self):
        super().__init__()

        self.num_classes = 6  # activities in the training set
        self.mapping = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.range(self.num_classes, dtype=tf.int32),
                values=[
                    "Walking",
                    "Jogging",
                    "Upstairs",
                    "Downstairs",
                    "Sitting",
                    "Standing",
                ],
            ),
            "Unknown",
        )

        self.num_features = 3  # sensor (x,y,z)
        self.batch_size = 32

        # 33,Jogging,49106062271000,5.012288,11.264028,0.95342433;
        self._model = k.Sequential(
            [
                k.layers.Input(
                    shape=(1, self.num_features), batch_size=self.batch_size
                ),
                # Note the stateful=True
                k.layers.LSTM(64, stateful=True),
                k.layers.Dense(self.num_classes),
            ]
        )

        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self._optimizer = k.optimizers.SGD(learning_rate=1e-4)
        # Sparse, so we can feed the scalar and get the one hot representation
        # From logits so we can feed the unscaled (linear activation fn)
        # directly to the loss
        self._loss = k.losses.SparseCategoricalCrossentropy(from_logits=True)

        self._last_tracked_activity = tf.Variable(-1, dtype=tf.int32, trainable=False)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 1, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ]
    )
    def learn(self, sensor_data, labels):
        # All the sensor data should be about the same activity
        tf.assert_equal(labels, tf.zeros_like(labels) + labels[0])

        # If the activity changes, we must reset the RNN state since the last update
        # and the current update are not related.

        if tf.not_equal(self._last_tracked_activity, labels[0]):
            tf.print(
                "Resetting states. Was: ",
                self._last_tracked_activity,
                " is ",
                labels[0],
            )
            self._last_tracked_activity.assign_sub(labels[0])
            self._model.reset_states()

        self._global_step.assign_add(1)
        with tf.GradientTape() as tape:
            loss = self._loss(labels, self._model(sensor_data))
            tf.print(self._global_step, ": loss: ", loss)

        gradient = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradient, self._model.trainable_variables))
        return loss

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 1, 3), dtype=tf.float32)])
    def predict(self, sensor_data):
        predictions = self._model(sensor_data)
        predicted = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        tf.print(self.mapping.lookup(predicted))
        return predicted


def main() -> int:
    at = ActivityTracker()

    # Executing an invocation of every graph we want to export is mandatory
    at.learn(
        tf.zeros((at.batch_size, 1, 3), dtype=tf.float32),
        tf.zeros((at.batch_size), dtype=tf.int32),
    )
    at.predict(tf.zeros((at.batch_size, 1, 3), dtype=tf.float32))

    tf.saved_model.save(at, "at")

    restored = tf.saved_model.load("at")
    return 0


if __name__ == "__main__":
    sys.exit(main())
