---
layout: post
title: "Dissecting tf.function: analyzing AutoGraph discovering its strength and subtlety"
date: 2019-04-15 08:00:00
categories: tensorflow
summary: "TODO"
---

AutoGraph is one of the most exciting new features of Tensorflow 2.0: it allows transforming a subset of Python syntax into its portable, high-performance and language agnostic graph representation bridging the gap between Tensorflow 1.x and the upcoming release based on the eager execution.

As often happens all that glitters is not gold: although powerful, AutoGraph hides some subtlety that is worth knowing; this article will guide you trough them using an hands-on approach.

## Session execution

The reader familiar with Tensorflow 1.x already knows that the standard workflow to get a callable graphs is:

1. Create the `tf.Graph` object and set it a default graph for the current scope.
2. Describe the computation using the Tensorflow API (e.g. `y = tf.matmul(A,x) + b`)
3. Create and configure a `tf.Session`
4. Build the concrete graph and put it into the `tf.Session`
5. Initialize all the variables
6. Use the `tf.Session.run` method to trigger the Graph execution, backtracking from the chosen node to execute to its inputs, and getting the result of the computation.

That translated to code looks like

```python
g = tf.Graph()
with g.as_default():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(A, x) + b
    init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(y)
```

Tensorflow 2.0, defaulting on eager execution follows a completely different approach based on the direct execution of what the user wants.

The bridge among the two worlds is `tf.function`.

## tf.function

In Tensorflow 2.0, AutoGraph is automatically applied to a function when it is decorated with `@tf.function`.

[`tf.function`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/function), in fact, creates callable graphs from Python functions. But how does it work?

Let's suppose we want to increase the performance of the matrix multiplication + variable addition defined in the previous section converting it to its graph representation.

Here are the steps to follow

### Functions, not Sessions

The major change of TF2 is the removal of the `tf.Session` object (see [RFC: Functions, not Sessions](https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md)). This change forces the user to organize the code in a better way: no more a `tf.Session` object to pass around, but just plain old python function that can be acceleated with a simple annotation.

Thus, we need to define a function that performs the desired task and annotate it with `@tf.function` to convert it in its graph representation and speed up the computation.

> Note: the speed-up is not always guaranteed. There are certain tasks in which is not worth converting the function to its correspondent graph, as is the case of this simple matrix multiplication we are performing here.
> However, for computationally intensive tasks like the optimization of a deep neural network the Graph conversion provides a huge performance boost.

```python
@tf.function
def mul():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(A, x) + b
    return y
```

