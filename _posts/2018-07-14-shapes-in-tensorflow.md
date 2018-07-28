---
layout: post
title: "Understanding Tensorflow's tensors shape: static and dynamic"
date: 2018-07-14 8:00:00
categories: tensorflow
summary: "Describing computational graphs is just a matter connecting nodes correctly. Connecting nodes seems a trivial operation, but it hides some difficulties related to the shape of tensors. This article will guide you through the concept of tensor's shape in both its variants: static and dynamic."
---

Describing computational graphs is just a matter of connecting nodes correctly. Connecting nodes seems a trivial operation, but it hides some difficulties related to the shape of tensors. This article will guide you through the concept of tensor's shape in both its variants: static and dynamic.

## Tensors: the basic

Every tensor has a name, a type, a rank and a shape.

- The **name** uniquely identifies the tensor in the computational graphs (for a complete understanding of the importance of the tensor name and how the full name of a tensor is defined, I suggest the reading of the article [Understanding Tensorflow using Go](/tensorflow/go/2017/05/29/understanding-tensorflow-using-go/)).
- The **type** is the data type of the tensor, e.g.: a `tf.float32`, a `tf.int64`, a `tf.string`, ...
- The **rank**, in the Tensorflow world (that's different from the mathematics world), is just the number of dimension of a tensor, e.g.: a scalar has rank 0, a vector has rank 1, ...
- The **shape** is the number of elements in each dimension, e.g.: a scalar has a rank 0 and an empty shape `()`, a vector has rank 1 and a shape of `(D0)`, a matrix has rank 2 and a shape of `(D0, D1)` and so on.

So you might wonder: what's difficult about the shape of a tensor? It just looks easy, is the number of elements in each dimension, hence we can have a shape of `()` and be sure to work with a scalar, a shape of `(10)` and be sure to work with a vector of size 10, a shape of `(10,2)` and be sure to work with a matrix with 10 rows and 2 columns. Where's the difficulty?

## Tensor's shape

The difficulties (and the cool stuff) arises when we dive deep into the Tensorflow peculiarities, and we find out that there's no constraint about the definition of the shape of a tensor. Tensorflow, in fact, allows us to represent the shape of a Tensor in 3 different ways:

1. **Fully-known shape**: that are exactly the examples described above, in which we know the rank and the size for each dimension.
2. **Partially-known shape**: in this case, we know the rank, but we have an unknown size for one or more dimension (everyone that has trained a model in batch is aware of this, when we define the input we just specify the feature vector shape, letting the batch dimension set to `None`, e.g.: `(None, 28, 28, 1)`.
3. **Unknown shape and known rank**: in this case we know the rank of the tensor, but we don't know any of the dimension value, e.g.: `(None, None, None)`.
4. **Unknown shape and rank**: this is the toughest case, in which we know nothing about the tensor; the rank nor the value of any dimension.

Tensorflow, when used in its non-eager mode, separates the graph definition from the graph execution. This allows us to first define the relationships among nodes and only after executing the graph.

When we define a ML model (but the reasoning holds for a generic computational graph) we define the network parameters completely (e.g. the bias vector shape is fully defined, as is the number of convolutional filters and their shape), hence we are in the case of a fully-known shape definition.

But a graph execution time, instead, the relationships among tensors (not among the network parameters, that remain constants) can be extremely dynamic.

To completely understand what happens at graph definition and execution time let's say we want to define a simple encoder-decoder network (that's the base architecture for convolutional autoencoders  / semantic segmentation networks / GANs and so on...) and let's define this in the more general possible way.

## encoder-decoder network architecture

This network accepts in input an image of any depth (1 or 3 channels) and with any spatial extent (height, width).
I'm going to use this network architecture to show you the concepts of static and dynamic shapes and how many information about the shapes of the tensors and of the network parameters we can get and use in both, graph definition time and execution time.

```python
inputs_ = tf.placeholder(tf.float32, shape=(None, None, None, None))
depth = tf.shape(inputs_)[-1]
with tf.control_dependencies([
        tf.Assert(
            tf.logical_or(tf.equal(depth, 3), tf.equal(depth, 1)), [depth])
]):
    inputs = tf.cond(
        tf.equal(tf.shape(inputs_)[-1], 3), lambda: inputs_,
        lambda: tf.image.grayscale_to_rgb(inputs_))

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
```

{% include inarticlead.html %}

This example hides some interesting features of Tensorflow's ops I/O shapes. Let's analyze in detail the shape of every layer, this will help us understand a lot about the shaping system.

## Dynamic input shape handling

A placeholder defined in this way

```python
inputs_ = tf.placeholder(tf.float32, shape=(None, None, None, None))
```

has an unknown shape and a known rank (4), at graph definition time.

At graph execution time, when we feed a value to the placeholder, the shape becomes fully defined: Tensorflow checks for us if the rank of the value we fed as input matches the specified rank and leave us the task to dynamically check if the passed value is something we're able to use.

So, this means that we have 2 different shapes for the input placeholder: a **static shape**, that's known at graph definition time and a **dynamic shape** that will be known only at graph execution time.

In order to check if the depth of the input image is in the accepted value (1 or 3) we have to use `tf.shape` and **not** `inputs_.shape`.

The difference between the `tf.shape` function and the `.shape` attribute is crucial:

- `tf.shape(inputs_)` returns a 1-D integer tensor representing the **dynamic shape** of `inputs_`.
- `inputs_.shape` returns a python tuple representing the **static shape** of `inputs_`.

Since the static shape known at graph definition time is `None` for every dimension, `tf.shape` is the way to go. Using `tf.shape` forces us to move the logic of the input shape handling inside the graph. In fact, if at graph definition time the shape was known, we could just use python and do something as easy as:

```python
depth = inputs_.shape[-1]
assert depth == 3 or depth == 1
if depth == 1:
    inputs_ = tf.image.grayscale_to_rgb(inputs_)
```

but in this particular case this is not possible, hence we have to move the logic inside the graph. The equivalent of the previous code defined directly into the graph is:

```python
depth = tf.shape(inputs_)[-1]
with tf.control_dependencies([
        tf.Assert(
            tf.logical_or(tf.equal(depth, 3), tf.equal(depth, 1)), [depth])
]):
    inputs = tf.cond(
        tf.equal(tf.shape(inputs_)[-1], 3), lambda: inputs_,
        lambda: tf.image.grayscale_to_rgb(inputs_))
```

from now on, we know that the input depth will be `3`, but Tensorflow **at graph definition time** is not aware of this (in fact, we described all the input shape control logic into the graph, and thus all of this will be executed only when the graph is created).

Created an input with a "well-known" shape (we do only know that the depth at execution time will be `3`) we want to define the encoding layer, that's just a set of 2 convolutional layers with a `3x3` kernel and a stride of `2`, followed by a convolutional layer with a kernel `6x6` and a stride of `1`.


But before doing this, we have to think about the variable definition phase of the convolutional layers: as we know from the [definition of the convolution operation among volumes](neural-networks/2016/11/24/convolutional-autoencoders/#convolution-among-volumes) in order to produce an **activation map** the operation needs to span all the input depth $$D$$.

This means that the *depth of every convolutional filter* depends on the input depth $$D$$, hence the variable definition depends on the expected input depth of the layers.

The shape of the variables **must always be defined** (otherwise the graph can't be built!).

*This means that we have to make Tensorflow aware at graph definition time of something that will be known only at graph execution time (the input depth).*

Since we know that after the execution of `tf.cond` the `inputs` tensor will have a depth of `3` we can use this information at graph definition time, **setting the static shape** to `(None,None,None,3)`: that's all we need to know to correctly define all the convolutional layers that will come next.

```python
inputs.set_shape((None, None, None, 3))
```

the `.set_shape` method simply assigns to the `.shape` property of the tensor the specified value.

In this way, the definition of all the convolutional layer `layer1`, `layer2` and `encode` can succeed. Let's analyze the shapes of the `layer1` (the same reasoning applies for every convolutional layer in the network):

## Convolutional layer shapes

At graph definition time we know the input depth `3`, this allows the `tf.layers.conv2d` operation to correctly define a set `32` convolutional filters each with shape `3x3x3`, where `3x3` is the spatial extent and the last `3` is the input depth (remember that a convolutional filter must span all the input volume).

Also, the `bias` tensor is added (a tensor with shape `(32)`).

So the input depth is all the convolution operation needs to know to be correctly defined (obviously, together with all the static information, like the number of filters and their spatial extent).

**What happens at graph execution time?**

The variables are untouched, their shape remains constant. Our convolution operation, however, spans not only the input depth but also all the input spatial extent (width and height) to produce the activation maps.

At graph definition time we know that the input of `layer1` will be a tensor with static shape `(None, None, None, 3)` and its output will be a tensor with static shape (`None, None, None, 32)`: nothing more.

*suggestion: just add a `print(layer)` after every layer definition to see every useful information about the output of a layer, including the static shape and the name.*

But we know that the output shape of a convolution can be calculated as (for both $$W$$ and $$H$$):

$$ O = \frac{W - 2K +P}{S} + 1 $$

This information can be used to add an additional check on the dynamic input shape, in fact, is possible to define a lower bound on the input resolution (a pleasure let to the reader).

## Decode layer shapes

As almost everyone today knows, the "deconvolution" operation produces chessboard artifacts[^1]. The standard solution is to replace the `tf.layers.conv2d_transpose` call to a `resize` + `conv2d` op.

In our case, this solution is the perfect representation on how we can create "mirror" architecture (the decoder is the mirror of the encoder), that will be able to correctly upsample the feature maps to the same exact shape of its twin.

Doing this using the information available at graph definition time (static shapes) is impossible. Instead, if we move everything inside the graph execution create this mirror architecture is extremely easy.

In fact, since `tf.shape` returns a 1-D tensor that represents the dynamic shape of its input, we can easily slice it in order to get only the spatial extent of the twin of the specified decoding layer.

In fact, the twin of the `layer2` can be defined easily in this way:

```python
d_layer2 = tf.image.resize_nearest_neighbor(encode, tf.shape(layer2)[1:3])
d_layer2 = tf.layers.conv2d(
	d_layer2,
	32,
	kernel_size=(3, 3),
	strides=(2, 2),
	activation=tf.nn.relu,
	padding="SAME",
	name="d_layer2")
```

At the first line, we resize the `encode` tensor to the spatial extent of `layer2`. After that, we just apply a standard convolution operation with padding `same` in order to maintain the same spatial extent.

That's all. Doing the same for every layer allows us to define the decoder as the mirror of the encoder (and this opens the road to other applications, like adding skip connections among twins, since the dimensions will always match, and so on...).


## Summary

- Variables must always be fully defined: exploit information from the graph execution time to correctly define meaningful variables shapes.
- There's a clear distinction between static and dynamic shapes: graph definition time and graph execution time must always be kept in mind when defining a graph.
- `tf.shape` allows defining extremely dynamic computational graphs, at the only cost to move the logic directly inside the graph and thus out of python.
- The resize operations accept dynamic shapes: use them in this way.

## Bonus: How to count the total number of trainable parameters in a Tensorflow model?

After reading this article, what's needed to count the total number of parameter in a Tensorflow model should be obvious.

We know that every variable must be fully defined, we can count the total number of parameter directly in the graph definition phase with a simply python loop, just accessing the `.shape` property of every trainable variable (`tf.trainable_variables()`).

For a complete answer, I let here a link to my StackOverflow answer to this question: https://stackoverflow.com/a/38161314/2891324

#### Why did I decided to write this post?

I see an increasing need in the community to understand how to correctly work with Tensorflow and its dynamic/static shape features. In fact, 2 of the most voted answer I wrote on StackOverflow are about this topic:

- https://stackoverflow.com/a/37085824/2891324 *tf.shape() get wrong shape in tensorflow*
- https://stackoverflow.com/a/38161314/2891324 *How to count total number of trainable parameters in a Tensorflow model?*

Probably the official documentation is not so clear about this aspect of the framework: I hope this post help to clarify this aspect.

If you find this article useful, feel free to share it using the buttons below!

---
[^1]: <a href="https://distill.pub/2016/deconv-checkerboard/">Deconvolution and Checkerboard Artifacts</a>
