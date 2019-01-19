---
layout: post
title: "Tensorflow 2.0: Keras is not (yet) a simplified interface to Tensorflow"
date: 2019-01-19 08:00:00
categories: tensorflow keras
summary: "In Tensorflow 2.0 Keras will be the default high-level API for building and training machine learning models, hence complete compatibility between a model defined using the old tf.layers and the new tf.keras.layers is expected. In version 2 of the popular machine learning framework the eager execution will be enabled by default although the static graph definition + session execution will be still supported. In this post, you'll see that the compatibility between a model defined using tf.layers and tf.keras.layers is not always guaranteed."
---

In Tensorflow 2.0 Keras will be the default high-level API for building and training machine learning models, hence complete compatibility between a model defined using the old `tf.layers` and the new `tf.Keras.layers` is expected. In version 2 of the popular machine learning framework the eager execution will be enabled by default although the static graph definition + session execution will be still supported (but hidden a little bit).

In this post, you'll see that the compatibility between a model defined using `tf.layers` and `tf.keras.layers` is not always guaranteed when using the graph definition + session execution, but it works as expected if the eager execution is enabled (at least from my tests).
1
The post is organized as follows: definition of the common data input pipeline, definition of the same model using both `tf.layers` and `tf.keras.layers`, analysis of different behaviors through 6 experiments.

## Input definition

The model we're going to use to highlight the differences between the 2 versions is a simple binary classifier. Instead of using a single `tf.data.Dataset` object with both the positive and negative classes inside, we want to use the variable sharing feature of Tensorflow (and thus of Keras models) to feed first the positive and then the negative ones in order to test also if the behavior of the variable sharing still works as we're used to.

```python
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
```

The dataset is simply an MNIST-like dataset of 50 black and 50 white images; the black images with label 0 and the white images with label 1. The two functions `get_positive` and `get_negative` return a batch with the desired number of elements.

## Model definition

The model is just a simple Convolutional Neural Network (CNN) with 2 convolutional layers, a batch normalization layer among them and a fully connected layer at the end with one output neuron used to discriminate between positive and negative inputs.

The batch normalization layer, due to how the batch normalization operation is defined, requires to update the moving mean and variance at its associated and this operation must be executed at every training step.

Defining the model using `tf.layers` is straight forward, in particular, there's only a way to define the input of this model and we have to use `tf.variable_scope` `reuse` parameter to share the model parameters.

Using Keras instead, we have 3 different ways of defining the input layer of the model, 2 of them are equivalent while the last one is completely different. Have a look at the next sections to understand the differences.

The goal of this evaluation is to show how the model behaves differently depending on how the input is defined and to highlight the problem that affects some Keras model, in particular their `updates` property.

### tf.layers

The definition is straightforward:

```python
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
```

There's no other way of defining this model. The input `feature` must be a `tf.Tensor` and due to `tf.AUTO_REUSE` the first call of `tf_model` will define the model and any other call will just reuse the model variables, just changing the input.

### Keras layers

```python
def keras_model(input_type):
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
```

Using Keras layers we have 3 options for defining the input layer:

- **shape**: specify the `input_shape` argument of the first layer: we know thus the exact output shape of every layer just after its definition;
- **layer**: define explicitly an input layer, where we specify the expected input shape; exactly as above, **shape** and **layer** ways are equivalent;
- **dynamic**: just declare a layer without specifying the expected input shape. Keras will take care of resolving the layer shapes when a new input is given to the model.

Of course, is clear that differently from `tf_model` the Keras function returns a Keras Model; if you're not familiar with the concept of Keras model or you're used to think in term of global graphs, I suggest you read the article [Tensorflow 2.0: models migration and new design](/tensorflow/gan/2018/11/04/tensorflow-2-models-migration-and-new-design/) that will guide you through the differences between the 1.x and 2.0 models definition and required change in your way of thinking.

## Experiments

So far, so good. We defined 2 input datasets and 2 functions able to define a Keras model / instantiate an output Tensor.

Now let's focus on the main topic of this article: find the differences between the two versions. For doing it, we'll run all these experiments:

1. Train the `tf.layers` model only on positives;
2. Train the `tf.layers` model on positive and negatives, hence test the variable sharing;
3. Train the Keras model defined using the **shape** of the input only on positives;
4. Train the Keras model defined using the **shape** of the input on positives and negatives;
5. Train the Keras model defined using the **dynamic** input shape only on positives;
6. Train the Keras model defined using the **dynamic** input shape on positives and negatives.

Training a binary classifier only on positive examples is meaningless, but for the purpose of this article is OK.

We run all these experiments keeping an eye on the values of the moving mean and variance that should be computed by the batch normalization layer. Moreover, since the most recent official Tensorflow documentation is all about eager execution, we give some suggestion about how to use the graph + session execution when using Keras layers to define the models (stuff learned the hard way, via a lot of trial and error).

In order to watch the moving mean and variance values we're going to use this rough helper function:

```python
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
```

### tf.layers experiments

We only have to evaluate two different cases, hence our main function used to run both experiments has the `case` input parameter.

```python
def main_layers(case):
    positive, positive_labels = get_positive(10)
    negative, negative_labels = get_negative(10)

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
```

The code is really easy and standard: we condition our `train_op` to run only after the `update_ops` defined in the global graph (`tf.get_default_graph()`): since we have a batch normalization layer, we're just making it work correctly.
When the function is invoked with `case == 2` we do variable sharing and train the model on positive and negatives, otherwise, we just train the model on the positives.

#### Experiment 1

The model trained only on positives. In short, this is a standard classifier architecture, hence we do expect that everything works. In fact, the output makes sense

```text
loss:  0.69315547
loss:  0.6928972
loss:  0.69264734
loss:  0.69239765
loss:  0.6921481
[array([0.01307349, 0.01343885, 0.02533867, 0.        , 0.00350613,
       0.        , 0.        , 0.015996  ], dtype=float32), array([0.95099014, 0.95099014, 0.95099014, 0.95099014, 0.95099014,
       0.95099014, 0.95099014, 0.95099014], dtype=float32)]
```

The moving mean and variance have been updated and everything works as expected.

#### Experiment 2

The model trained on positive and negatives samples sharing variables. As in the previous experiment, we expect everything to go smoothly, in fact:

```text
loss:  1.3862944
loss:  1.3862944
loss:  1.3862944
loss:  1.3862944
loss:  1.3862944
[array([0.        , 0.00704158, 0.00868269, 0.        , 0.        ,
       0.        , 0.        , 0.0007124 ], dtype=float32), array([0.90438217, 0.90438217, 0.90438217, 0.90438217, 0.90438217,
       0.90438217, 0.90438217, 0.90438217], dtype=float32)]
```

Conclusion: having a straightforward way of defining a model and sharing variables makes everything goes smoothly. Let's see if the 2 different ways of making the same with Keras goes as smooth as the traditional one.

### Keras layer experiments

In order to run the 4 Keras experiments we define this function:

```python
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

    with tf.control_dependencies(model.updates):
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

```

**NOTE**: right now the Tensorflow tutorial and documentation are all focused on the eager-execution and therefore all the examples there presented miss the most important part for the correct graph + session execution, the `tf.keras.backend.set_learning_phase(True)`.
This sets the Keras status to training. One may argue that there's the `model.trainable` property that can be set to `True`. Just try to remove `tf.keras.backend.set_learning_phase(True)` and add `model.trainable = True`: it doesn't work. In graph + session execution that property is completely ignored!

Another difference between the `tf.layers` solution and this one is the lack of global collections, hence every model carries its own update operations in the `.updates` property.

#### Experiment 3

Train the Keras model defined using the **shape** of the input only on positives; we expect everything to work correctly since is a standard classifier.

**SURPRISE**: it fails.

The output in fact is:

```text
CASE:  1
INPUT TYPE:  shape
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 13, 13, 8)         80
_________________________________________________________________
batch_normalization (BatchNo (None, 13, 13, 8)         32
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 6, 6, 1)           73
_________________________________________________________________
flatten (Flatten)            (None, 36)                0
_________________________________________________________________
dense (Dense)                (None, 1)                 37
=================================================================
Total params: 222
Trainable params: 206
Non-trainable params: 16
_________________________________________________________________

update_ops:  []
model.updates:  [<tf.Operation 'batch_normalization/AssignMovingAvg/AssignSubVariableOp' type=AssignSubVariableOp>, <tf.Operation 'batch_normalization/AssignMovingAvg_1/AssignSubVariableOp' type=AssignSubVariableOp>]

[...]

 You must feed a value for placeholder tensor 'conv2d_input' with dtype float and shape [?,28,28,1]
```

As we can see the model summary prints correctly every info about our model, but at the first `sess.run` it fails. What's going on? We correctly defined the model, there's no variable sharing and our input comes from a `tf.data.Dataset` object, it should work everything as smoothly as in the `tf.layers` experiment!

In practice, what's happening is when we define the input shape in the Keras model, the model defines an input placeholder that needs to be fed when the model output tensor is executed.
But the `model.updates` have been defined using a new and different input placeholder that *we are not able to feed*.

To verify this, we can just comment the line `with tf.control_dependencies(model.updates):`. If we run the train operation we get:

```text
loss:  0.6931472
loss:  0.6928972
loss:  0.69264734
loss:  0.69239765
loss:  0.6921481
[array([0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32), array([1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32)]

```
The model trains but the update operations are not executed therefore there's no way to save the moving mean and variable computed for every batch by the batch normalization layer: this makes every operation that has its own update operation completely useless.

### Experiment 4

Train the Keras model defined using the **shape** of the input on positives and negatives.

This experiment has the same behavior of experiment 3. Thus when there are update operations to execute, no matter if variable sharing is used or not, the problem with the unusable input of the update operation arises.

Removing the execution of `model.updates` (hence removing the `tf.control_dependencies`) has the same behavior of experiment 3: the model trains, but the batch normalization layer are useless.

Moreover, when trying to execute just the line `sess.run(model.updates)` (and not the `train_op`) the behavior is the same.

In my opinion, this behavior is completely wrong, since I defined the input of my model using `tf.data`, I never created a placeholder and thus when I execute an operation that requires the input data it should be fetched from the `tf.data.Dataset` object I defined and not from a placeholder I never defined.

However, update operations like these works correctly when using the Keras `Model` methods `.fit` or `.train_on_batch` (just to increase the confusion): Keras layers are not a drop-in replacement for the Tensorflow layers, but they work correctly when used inside the Keras framework.

Perhaps Keras can't be used as a [Simplified interface to Tensorflow](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html) as claimed in the Keras blog?

Let's continue with the last two experiments.

### Experiments 5

Train the Keras model defined using the **dynamic** input shape only on positives. This is just a simple classifier architecture, exactly as the one created in experiments 3.
The architecture defined in experiment 3 defines the input shape directly in the input layer, while this one becomes aware of the input dimensions only after instantiating the model.

However, since experiment 3 failed we do expect this experiment to fail too.

**SURPRIRE**: it works.

Here's the output:

```text
CASE:  1
INPUT TYPE:  dynamic
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              multiple                  80
_________________________________________________________________
batch_normalization (BatchNo multiple                  32
_________________________________________________________________
conv2d_1 (Conv2D)            multiple                  73
_________________________________________________________________
flatten (Flatten)            multiple                  0
_________________________________________________________________
dense (Dense)                multiple                  37
=================================================================
Total params: 222
Trainable params: 206
Non-trainable params: 16
_________________________________________________________________
update_ops:  []
model.updates:  [<tf.Operation 'sequential/AssignMovingAvg/AssignSubVariableOp' type=AssignSubVariableOp>, <tf.Operation 'sequential/AssignMovingAvg_1/AssignSubVariableOp' type=AssignSubVariableOp>]
2019-01-19 16:45:24.995180: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
loss:  0.6931472
loss:  0.6928972
loss:  0.69264734
loss:  0.69239765
loss:  0.6921481
[array([0.00394114, 0.04436748, 0.05262341, 0.02130721, 0.00692791,
       0.00136741, 0.        , 0.        ], dtype=float32), array([0.95099014, 0.95099014, 0.95099014, 0.95099014, 0.95099014,
       0.95099014, 0.95099014, 0.95099014], dtype=float32)]
```

The update operations instantiated in this way use the input provided, without defining a new input placeholder. The problem of the dynamic shape is the loss of the output shape information for every layer of the model. In fact, in the summary, we can see that instead of a tuple with the output shape, we have the value "multiple".

Why not defining the input layer, and thus losing the static shape information, we have the correct behavior?

### Experiment 6

Train the Keras model defined using the **dynamic** input shape on positives and negatives.

Here we go, a model defined using the **dynamic** input shape works exactly like the model defined using the `tf.layers`, in fact, the output is:

```text
CASE:  2
INPUT TYPE:  dynamic
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              multiple                  80
_________________________________________________________________
batch_normalization (BatchNo multiple                  32
_________________________________________________________________
conv2d_1 (Conv2D)            multiple                  73
_________________________________________________________________
flatten (Flatten)            multiple                  0
_________________________________________________________________
dense (Dense)                multiple                  37
=================================================================
Total params: 222
Trainable params: 206
Non-trainable params: 16
_________________________________________________________________
update_ops:  []
model.updates:  [<tf.Operation 'sequential/AssignMovingAvg/AssignSubVariableOp' type=AssignSubVariableOp>, <tf.Operation 'sequential/AssignMovingAvg_1/AssignSubVariableOp' type=AssignSubVariableOp>, <tf.Operation 'sequential_1/AssignMovingAvg/AssignSubVariableOp' type=AssignSubVariableOp>, <tf.Operation 'sequential_1/AssignMovingAvg_1/AssignSubVariableOp' type=AssignSubVariableOp>]
2019-01-19 16:50:44.534061: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
loss:  1.3862879
loss:  1.3862879
loss:  1.3862879
loss:  1.3862879
loss:  1.3862879
[array([0.        , 0.        , 0.        , 0.02384297, 0.02239617,
       0.01027304, 0.00340564, 0.00223023], dtype=float32), array([0.90392077, 0.90392077, 0.90392077, 0.90392077, 0.90392077,
       0.90392077, 0.90392077, 0.90392077], dtype=float32)]
```

As we can see variable sharing has been correctly implemented and the 2 different inputs using the same model implied the creation of the correct number of update operations for every input.

The numerical results are comparable with the one obtained with `tf.layers`, therefore using this way of defining the models is the correct way to go.

## Conclusion

There are still to many difference in how Tensorflow behaves and how a generic framework like Keras has been designed to work: Keras model should be trained using `model.fit` or `model.train_on_batch` and the user should let Keras take care of everything.
Since Keras is going to be the new interface for defining the model on Tensorflow 2.0 and there are uses that simply don't want to use the training abstraction offered by Keras (or because they're researching new way of training, or because in their use case the standard training procedure can't just work, ...) all these differences, in my opinion, needs to disappear.
The behavior of `tf.keras.layers` should be *always* the same as `tf.layers`, no matter how I defined the input of the model.

Thus, using Keras as a simplified interface to Tensorflow is more or less a lie, at least if we want to use the graph definition + session execution. The behavior and the whole structure of the model change depending on how we define the input layer.

Moreover, focusing all the new documentation on the eager execution + Keras (that in this combo just works) there are practically zero examples on how to use the graph definition using the Keras layers + a proper execution inside a Session. Probably this is due to the fact that in Tensorflow 2.0 `tf.Session` will disappear and we have to learn how to use `@tf.function` that will define a graph and a session execution for us; see [RFC: Functions, not Sessions](https://github.com/tensorflow/community/blob/2b712e59cf572ccf4c463519b0e062ad3c48bbe8/rfcs/20180918-functions-not-sessions-20.md).

### Info, code, and reference

- If you want to download the rough script used to run the experiments: [download it here](/code/keras-input/test.py).
- I have opened an [issue on GitHub](https://github.com/tensorflow/tensorflow/issues/23873) about this some time ago: being unable to correctly migrate my models from `tf.layers` to `tf.keras.layers` I decided to debug and analyze the problem by myself. This blog post is the result of my debugging.
- [RFC: Functions, not Sessions](https://github.com/tensorflow/community/blob/2b712e59cf572ccf4c463519b0e062ad3c48bbe8/rfcs/20180918-functions-not-sessions-20.md).
- [Keras as a simplified interface to Tensorflow](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html)
- [Tensorflow 2.0: models migration and new design](/tensorflow/gan/2018/11/04/tensorflow-2-models-migration-and-new-design/)


If you liked the article feel free to share it using the buttons below and don't hesitate to comment to let me know if there's something wrong/that can be improved in the article.

Thanks for reading!
