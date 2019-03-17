---
layout: post
title: "Dissecting tf.function: analyzing AutoGraph discovering its strength and subtlety"
date: 2019-03-15 08:00:00
categories: tensorflow
summary: "AutoGraph is one of the most exciting new features of Tensorflow 2.0: it allows transforming a subset of Python syntax into its portable, high-performance and language agnostic graph representation bridging, in this way, the gap between Tensorflow 1.x and the 2.0 release based on eager execution. As often happens all that glitters is not gold: although powerful, AutoGraph hides some subtlety that is worth knowing; this article will guide you trough them using an hands-on approach."
---

AutoGraph is one of the most exciting new features of Tensorflow 2.0: it allows transforming a subset of Python syntax into its portable, high-performance and language agnostic graph representation bridging, in this way, the gap between Tensorflow 1.x and the 2.0 release based on eager execution.

As often happens all that glitters is not gold: although powerful, AutoGraph hides some subtlety that is worth knowing; this article will guide you trough them using an hands-on approach.

## Session execution

The reader familiar with Tensorflow 1.x already knows that the standard workflow to get a callable graphs is (or better, define a graph with nodes that can be executed within a `tf.Session`):

1. Create the `tf.Graph` object and set it as the default graph for the current scope.
2. Describe the computation using the Tensorflow API (e.g. `y = tf.matmul(a,x) + b`).
3. Create and configure the `tf.Session`.
4. Build the concrete graph and put it into the `tf.Session`.
5. Initialize all the variables.
6. Use the `tf.Session.run` method to trigger the Graph execution. The node execution will trigger a backtracking procedure from the chosen node to to its inputs, in order to resolve the dependencies and return the result.

All these 6 points can be translated in code with this minimal example:

```python
g = tf.Graph()
with g.as_default():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(y)
```

Tensorflow 2.0, defaulting on eager execution follows a completely different approach based on the direct execution of what the user wants.

- Remove the graph definitio
- Remove the session execution
- Remove the variables initialzation

just write the code and run it:

```python
a = tf.constant([[10,10],[11.,1.]])
x = tf.constant([[1.,0.],[0.,1.]])
b = tf.Variable(12.)
y = tf.matmul(a, x) + b
print(y.numpy())
```

The eager counterpart of any Tensorflow 1.x source code is usually slower, since it relies on the Python interpreter to run the computation and there are a lot of optimization that are only possibile on DataFlow graphs.

The bridge among the two worlds, that allow to create graphs even in Tensorflow 2.0 in order to have high-performance code, is `tf.function`.

## tf.function, not tf.Session

One of the major changes in Tensorflow 2.0 is the removal of the `tf.Session` object (see [RFC: Functions, not Sessions](https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md)). This change forces the user to organize the code in a better way: no more a `tf.Session` object to pass around, but just plain old Python function that can be accelerated with a simple decoration.

Thus, we need to define a function that performs the desired task and decorate it with `@tf.function` to convert it in its graph representation and speed up the computation.

> Note: the speed-up is not always guaranteed. There are certain tasks in which is not worth converting the function to its correspondent graph, as is the case of this simple matrix multiplication we are performing here.
> However, for computationally intensive tasks like the optimization of a deep neural network the Graph conversion provides a huge performance boost.

The automatic conversion from Python code to its graph representation is made by the AutoGraph module.

In Tensorflow 2.0, AutoGraph is automatically applied to a function when it is decorated with `@tf.function`. [`tf.function`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/function), in fact, creates callable graphs from Python functions. But how does it work?

The next sections will guide you trough the required steps to migrate a 1.x snippet to its eager and graph-accelerated 2.0 version.

## Update to eager execution

To work with `tf.function` the first thing to do is to refactor the old 1.x code, in order to make it modular. Where first was a session execution, now there should be a Python function.

```python
def f():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    return y
```

What happens now? **Nothing**. Tensorflow 2.0 works in eager mode by default, this mean that we just defined a standard python function and if we evaluate it:

```python
print(f().numpy())
```

We get as expected:
```
[[22. 22.]
 [23. 13.]]
```

## @tf.function decoration: first failure

Let's just add the `@tf.function` decoration to the `f` function. For the sake of clarity (and to debug in the old-school print driven way) let's add even a `print` and a `tf.print` statement inside the function body. The code now is

```python
@tf.function
def f():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y

f()
```
What happens now?

1. The annotation `@tf.function` wrapped the `f` function in a `tensorflow.python.eager.def_function.Function` object. The python function is assigned to the `.python_function` property of the object.
2. Until the object is called ( `f()` ): nothing happens.
3. When `f()` is called the process of graph building starts. At this stage, only the python code is executed and the behavior of the function is traced, in order to collect the required data to build the graph. Thus the only output we get is:
```
PRINT:  Tensor("add:0", shape=(2, 2), dtype=float32)
```
The `tf.print` call is not evaluated as any other `tf.*` method, since Tensorflow already knows everything about that statements and it can use them as they are to build the graph.
4. **FAIL**: during the first and only invocation of the function, the following exception has been raised
```
ValueError: tf.function-decorated function tried to create variables on non-first call.
```
`@tf.function` failed to build the graph.

In my opinion this is a bug and I've opened an issue. I should be able to declare variables inside functions decorated with `tf.function`: moreover, the message is not clear, since I call the function once, thus this is the first call, but the message is about a non-first call.

As a workaround, we can just declare the `f` function as a function that accepts an input parameter `b`.

## @tf.function decoration: function input parameters

In order to workaround the problem of the variable declaration inside the function body, we can refactor the `f` function in order that it accepts `b` as input parameter.

```python
@tf.function
def f(b):
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    y = tf.matmul(a, x) + b
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y

b = tf.Variable(12.)
f(b)
```

What happens? Points 1,2,3 are identical to the previous list. But now there is no failure:

The evaluation of `f(b)` produces the following output

```
PRINT:  Tensor("add:0", shape=(2, 2), dtype=float32)
TF-PRINT:  [[22 22]
 [23 13]]
```

The first `PRINT` is the Tensor, Tensorflow 1.x style (thus, no value, just the computation description as it happens in Tensorflow 1.x), that is used to build the computational graph.
The second line, `TF-PRINT` is the result of the session evaluation of the graph defined.
