---
layout: post
title: "Keras is not (yet) a simplified interface to Tensorflow"
date: 2019-01-19 08:00:00
categories: tensorflow keras
summary: ""
---

In Tensorflow 2.0 Keras will be the default high-level API for building and training machine learning models, hence a complete compatibility between a model defined using the old `tf.layers` and the new `tf.keras.layers` is expected. In the version 2 of the popular machine learning framework the eager execution will be enabled by default although the static graph definition + session exectution is still supported

In this post you'll see that the compatibility between a model defined using `tf.layers` and `tf.keras.layers` is not always guaranteed when using the graph definition + session execution, but it works as expected only if the eager execution is enabled.

The post is organized as follows: definition of the common data input pipeline, definition of the same model using both `tf.layers` and `tf.keras.layers`, analysis of the different behavirous.

The article focus on the static graph definition + session execution, since the eager execution works as expected (at least from my tests).

## Input definition

The model we're going to use to highlight the differences between the 2 versions is a simple binary classifier. Instead of using a single `tf.data.Dataset` object with both the positive and negative classes inside, we want to use the variable sharing feature of tensorflow (and thus of Keras models) to feed first the positive and then the negative ones: this behaviour in particular will show one of the biggest problem with the new implementation.

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

The dataset is simply a MNIST-like dataset of 50 black and 50 white images; the black images with label 0 and the white images with label 1. The two functions 'get_positive` and `get_negative` return a batch with the desired number of elements.

## Model definition

The model is just a simple Convolutional Neural Network (CNN) with 2 convolutional layer, a batch normalization layer among them and a fully connected layer at the end with one output neuron used to discriminate between positive and negative inputs.

The batch normalization layer, due to how the batch normalization operation is defined, requires to update of the moving mean and variance at its associated and this operation must be exected at every training step.

Defining the model using `tf.layers` is straight forward, in particular there's only a way to define the input of this model and we have to use `tf.variable_scope` `reuse` parameter to share the model parameters.

Using Keras instead, we have 3 different ways of defining the input layer of the model, 2 of them are equivalent while the last one is completely different. Have a look a the next sections to understand the differences.

The goal of this evaluation is to show how the model behaves differently depending on how the input is defined and to highlight the problem that affetcs any keras model, about its `updates` property.

### tf.layers

The defintion is straightforward:

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

There's no other way of defining this model. The input `feature` must be a `tf.Tensor` and the due to `tf.AUTO_REUSE` the first call of `tf_model` will define the model and any other call will just reuse the model variables, just changeing the input.

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

- **shape**: specify the `input_shape` argument of the first layer: we know thus the exact shape of any output tensor of the model just after its defnition;
- **layer**: define explictly an input layer, where we specify the expected input shape; exactly as above, **shape** and **layer** ways are equivalent;
- **dynamic**: just declare a layer without specify the expected input shape. Keras will take care of resolving the layer shapes when a new input is given to the model.

Of course, is clear that differently from `tf_model` the Keras function returns a Keras Model; if you're not familiar with the concet of keras model or you're used to think in term of global graphs, I suggest you reading the article [Tensorflow 2.0: models migration and new design](tensorflow/gan/2018/11/04/tensorflow-2-models-migration-and-new-design/) that will guide you trough the differences between the 1.x and 2.0 models defintion and required way of thinking change.

## Experiments

So far, so good. We defined 2 input datasets and 2 functions able to define a Keras model / instantiate an output Tensor.

Now let's focus on the main topic of this article: find the differences between the two versions. For doing it, we'll run all these experiments:

1. Train the `tf.layer` model only on positives;
2. Train the `tf.layer` model on positive and negatives, hence test the variable sharing;
3. Train the Keras model defined using the **shape** of the input only on positives;
4. Train the Keras model defined using the **shape** of the input on positives and negatives;
5. Train the Keras model defined using the **dynamic** input shape only on positives;
6. Train the Keras model defined using the **dynamic** input shape on positives and negatives.

Ttraining a binary classifier only on positive examples is meaningless, but for the purpose of this article is OK.

We run all these experiments keeping an eye on the values of the moving mean and variance that should be computed by the batch normalization layer. Moreover, since the most recent official Tensorflow documentation is all about eager execution, we give some suggestion about how to use the graph + session execution when using Keras (stuff learned the hard way, via a lot of trial and error).

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
When the function is invoked with `case == 2` we do variable sharing and train the model on positive and negatives, otherwise we just train the model on the positives.

#### Experiment 1

Model trained only on positives. In short, this is a standard classifier architeture, hence we do expect that everything works. In fact, the output makes sense

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

Model trained on positive and negatives samples sharing model variables. As in the previous experiment we expect everything to go smoothly, in fact:

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

Conclusion: havign a straightfoward way of definig a model and sharing variables makes everything goes smoothly. Let's see if the 2 different ways of making the same with Keras goes as smooth as the traditional one.

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

**NOTE**: right now the Tensorflow tutorial and documentation is all focused on the eager-execution and therefore all the examples there presented miss the most important part for the correct graph + session execution, the `tf.keras.backend.set_learning_phase(True)`.
This sets the Keras status to training. One may argue that there's the `model.trainable` property that can be set to `True`. Just try to remove `tf.keras.backend.set_learning_phase(True)` and add `model.trainable = True`: it doesn't work. In graph + session execution that property is completely ignored!

Another difference between the `tf.layers` solution and this one is the lack of global collection, hence every model carries its own update operations in the `.updates` property.


#### Experiment 3

Train the Keras model defined using the **shape** of the input only on positives; we expect everything to work correctly, since is a standard classifier.
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

What's going on? We correctly defined the model, there's no variable sharing and our input comes from a `tf.data.Dataset` object, it should work everything as smoothly as in the `tf.layer` experiment!

In practice, what's happening is when we define the input shape in the Keras model, the model defines an input placeholder that need to be fed when the model output tensor is executed.
But the `model.updates` have been defined using a different input, hence different placeholders,  that *we are not able to feed*.

To verify this, we can just comment the line `with tf.control_dependencies(model.updates):`. If we run train operation we get:

```text
loss:  0.6931472
loss:  0.6928972
loss:  0.69264734
loss:  0.69239765
loss:  0.6921481
[array([0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32), array([1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32)]

```
The model trains but the update operations are not executed therefore there's no way to save the moving mean and variable computed for every batch by the batch normalization layer.



## Conclusion

This article has been created with the specific aim of shed a light on the changes and the challenges that Tensorflow 2.0 will bring to us, the framework users.

The GAN implementation in Tensorflow 1.x and its conversion in Tensorflow 2.x should be a clear example of the mindset change required to work with the new version.

Overall I think Tensorflow 2.x will improve the quality of the framework and it will standardize and simplifies how to use it. New users that never seen a static-graph approach and are used to work with imperative languages could find the eager mode a good entry point to the Tensorflow world.

However, there are certain parts of the update that I don't like (please not that those are just my personal opinions):

- The focus on the eager execution and make it the default: it looks too much a marketing move to me. It looks like Tensorflow wants to chase PyTorch (eager by default)
- The missing 1:1 compatibility with static-graph and eager (and the possibility of mixing them) could create a mess in big projects IMHO: it would be hard to maintain this projects
- Switching to a Keras based approach is a good move, but it makes the graph visualized in Tensorboard *really* ugly. In fact, the variables and the graphs are defined globally, and the `tf.named_scope` (invoked every time a Keras Model is called, in order to share the variables easily) that creates a new "block" in the Tensorflow graph, is separated by the graph it uses internally and it has in the list of the input nodes all the variables of the model - this makes the graph visualization of Tensorboard pretty much useless and that's a pity for such a good tool.

If you liked the article feel free to share it using the buttons below and don't hesitate to comment to let me know if there's something wrong/that can be improved in the article.

Thanks for reading!
