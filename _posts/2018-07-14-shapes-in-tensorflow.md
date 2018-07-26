---
layout: post
title: "Shapes in tensorflow: dynamic vs static shape"
date: 2018-07-14 8:00:00
categories: tensorflow
summary: 
---

Describing a computational graphs is just a matter of correctly connecting nodes. Connecting nodes seems a trivial operation, but it hides some difficult related to the shape of tensors. This article will guide you trough the concept of tensor's shape in both its variants: static and dynamic.

## Tensors: the basic

Every tensor has a name, a type, a rank and a shape.

- The **name** uniquely identifies the tensor in the computational graphs (for a complete understanding of the importance of the tensor name and how the full name of a tensor is defined, I suggest the reading of the article [Understanding Tensorflow using Go](/tensorflow/go/2017/05/29/understanding-tensorflow-using-go/)).
- The **type** is the data type of the tensor, e.g.: a `tf.float32`, a `tf.int64`, a `tf.string`, ...
- The **rank**, in the Tensorflow world (that's different from the mathematics world), is just the number of dimension of a tensor, e.g.: a scalar has rank 0, a vector has rank 1, ...
- The **shape** is the number of elements in each dimension, e.g.: a scalar has a rank 0 and an empty shape `()`, a vector has rank 1 and a shape of `(D0)`, a matrix has rank 2 and a shape of `(D0, D1)` and so on.

So you might wonder: what's difficult about the shape of a tensor? It just looks easy, is the number of elements in each dimension, hence we can have a shape of `()` and be sure to work with a scalar, a shape of `(10)` and be sure to work with a vector of size 10, a shape of `(10,2)` and be sure to work with a matrix with 10 rows and 2 columns. Where's the difficult?

## Tensor's shape

The difficulties (and the cool stuff) arises when we dive deep into the Tensorflow peculiarities, and we find out that there's no constraint about the definition of the shape of a tensor. Tensorflow, in fact, allow us to represent the shape of a Tensor in 3 different ways:

1. **Fully-known shape**: that are exactly the examples described above, in which we know the rank and the size for each dimension.
2. **Partially-known shape**: in this case, we know the rank, but we have an unknown size for one or more dimension (everyone that has trained a model in batch is aware of this, when we define the input we just specify the feature vector shape, letting the batch dimension set to `None`, e.g.: `(None, 28, 28, 1)`.
3. **Unknown shape and known rank**: in this case we know the rank of the tensor, but we don't know any of the dimension value, e.g.: `(None, None, None)`.
4. **Unknown shape and rank**: this is the most tough case, in which we don't know nothing about the tensor; the rank nor the value of any dimension.

Tensorflow, when used in its non-eager mode, separates the graph definition from the graph execution. This allow us to first define the relationships among nodes and only after executing the graph.

When we define a ML model (but the reasoning holds for a generic computational graph) we define the network parameters completely (e.g. the bias vector shape is fully defined, as is the number of convolutional filter and their shape), hence we are in the case of a fully-known shape definition.

But a graph execution time, instead, the relationships among tensors (not among the network parameters, that remain constants) can be extremely dynamic.

To completely understand what happens at graph definition and execution time let's say we want to define a simple encoder-decoder network (that's the base architecture for convolutional autoencoders  / semantic segmentation networks / GANs and so on...) and let's define this in the more general possible way.

### encoder-decoder network architecture

This network accepts in input an image of any depth (1 or 3 channels) and with any spatial extent (height, width).
I'm going to use this network architecture to show you the concepts of static and dynamic shapes and how many information about the shapes of the tensors and of the network parameters we can get and use in both, graph definition time and execution time.

```python
inputs_ = tf.placeholder(tf.float32, shape=(None, None, None, None))
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

In this short example, we have the definition of an input **placeholder with a partially-known shape** and, for every `conv2d` layer, the definition of 2 **variables with a fully-known shape**.
In fact, every time we define a convolutional layer with `tf.layers.conv2d` we're defining both a `bias` vector and a `weights` tensor.

This example, however, hides some cool features of tensorflow's operations I/O shapes. Let's analyze in detail the shape of every layer, this will help us understand a lot about the shaping system.

### Input placeholder

As said above, the input placeholder shape is partially known, in fact we're explicitly telling to tensorflow that the first dimension (the batch dimension) is unknown, hence at graph execution time this value can be anything >= 1.

So, this means that we have 2 different shapes for the input placeholder: a **static shape**, that's known at graph definition time and a **dynamic shape** that will be known only at graph execution time (when we feed a value to that input placeholder).


