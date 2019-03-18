---
layout: post
title: "Dissecting tf.function: analyzing AutoGraph to discover its strength and subtleties"
date: 2019-03-15 08:00:00
categories: tensorflow
summary: "AutoGraph is one of the most exciting new features of Tensorflow 2.0: it allows transforming a subset of Python syntax into its portable, high-performance and language agnostic graph representation bridging the gap between Tensorflow 1.x and the 2.0 release based on eager execution. As often happens all that glitters is not gold: although powerful, AutoGraph hides some subtlety that is worth knowing; this article will guide you trough them."
---

AutoGraph is one of the most exciting new features of Tensorflow 2.0: it allows transforming a subset of Python syntax into its portable, high-performance and language agnostic graph representation bridging the gap between Tensorflow 1.x and the 2.0 release based on eager execution.

As often happens all that glitters is not gold: although powerful, AutoGraph hides some subtlety that is worth knowing; this article will guide you trough them.

## Session execution

The reader familiar with Tensorflow 1.x already knows that the standard workflow to get a callable graphs is (or better, define a graph with nodes that can be executed within a `tf.Session`):

1. Create the `tf.Graph` object and set it as the default graph for the current scope.
2. Describe the computation using the Tensorflow API (e.g. `y = tf.matmul(a,x) + b`).
3. Create and configure the `tf.Session`.
4. Build the concrete graph and load it into the `tf.Session`.
5. Initialize all the variables.
6. Use the `tf.Session.run` method to start the computation. The node execution will trigger a backtracking procedure from the chosen nodes (`.run` input parameters) to to their inputs, in order to resolve the dependencies and compute the result.

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

- Remove the graph definition.
- Remove the session execution.
- Remove the variables initialization.

Just write the code and run it:

```python
a = tf.constant([[10,10],[11.,1.]])
x = tf.constant([[1.,0.],[0.,1.]])
b = tf.Variable(12.)
y = tf.matmul(a, x) + b
print(y.numpy())
```

The eager counterpart of any Tensorflow 1.x source code is usually slower, since it relies on the Python interpreter to run the computation and there are a lot of optimization that are only possibile on DataFlow graphs.

The bridge among the two versions that allow creating computational graphs even in Tensorflow 2.0 is `tf.function`.

## tf.function, not tf.Session

One of the major changes in Tensorflow 2.0 is the removal of the `tf.Session` object (see [RFC: Functions, not Sessions](https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md)). This change forces the user to organize the code in a better way: no more a `tf.Session` object to pass around, but just plain old Python functions that can be accelerated with a simple decoration.

In order to define a graph in Tensorflow 2.0 we need to define a Python function that describes the computation and decorate it with `@tf.function`.

> **Note**: the speed-up is not guaranteed. There are certain tasks in which is not worth converting the function to its graph representation, as is the case of this simple matrix multiplication we are performing here.
> However, for computationally intensive tasks like the optimization of a deep neural network the Graph conversion provides a huge performance boost.

The automatic conversion from Python code to its graph representation is called AutoGraph.

In Tensorflow 2.0, AutoGraph is automatically applied to a function when it is decorated with [@tf.function](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/function); this decorator creates callable graphs from Python functions.

How does it work?

### tf.function: layman explanation

Long story short:

On the first call of a decorated function:

- The function body is executed and its behavior traced. Eager execution is disabled in this context, therefore every `tf.` method just define a `tf.Operation` node that produces a `tf.Tensor` output, Tensorflow 1.x like.
- AutoGraph is used to to detect python constructs that can be converted to their graph equivalent (`while` → `tf.while`, `for` → `tf.while`, `if` → `tf.cond`, `assert` → `tf.assert`, ...).
- From the function trace + autograph the graph representation of the function is built. In order to preserve the execution order in the defined graph, `tf.control_dependencies` is automatically added after every statement, in order to condition the line $$i+1$$ on the execution of the line $$i$$.
- The `tf.Graph` object has now been built.
- Based on the function name and the input parameters a unique ID is created and associated with the graph. The graph is cached into a map: `map[id] = graph`.
- Any function call will just re-use the defined graph if the key matches.

The next sections will guide you trough the required steps to migrate a 1.x snippet to its eager and graph-accelerated 2.0 version. During the conversion an analysis of the `tf.function` behavior is performed.

## Conversion to eager execution

To use `tf.function` the first thing to do is to refactor the old 1.x code, wrapping the code we want to execute into a session.

In general, where first there was a session execution, now there is Python function (**note**: this is a huge advantage - the software architecture is cleaner, easy to maintain and document).

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

We get the expected result:
```
[[22. 22.]
 [23. 13.]]
```

## tf.function: a failure and the need to refactor <small>(if the function creates a state)</small>

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

In my opinion this is a bug and [I've opened an issue](https://github.com/tensorflow/tensorflow/issues/26812).

The [RFC: Functions, not Session](https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md#functions-that-create-state) in the section dedicated to the functions that create a state clearly states:

> State (like tf.Variable objects) are only created the first time the function f is called.

Thus, I do expect a workflow like:

**First call**:

```python
f()
```

Graph definition and execution since this is the first time the function f is called.

**Any other call**:

```python
f() #again
```
Failure:

```
ValueError: tf.function-decorated function tried to create variables on non-first call.
```

What happens right now is not clear to me. I called the function once, but the message is about a non-first call - that's strange. Probably under the hood `tf.function` called twice the function definition (why?) but this makes the previously quoted statement from the RFC false.

Everyone agrees that the function call is creating a graph and is defining a `tf.Variable` inside. Hence, in my opinion, at the first call it should just define the `tf.Variable` object while any non-first call should fail if I don't take care of preventing the re-definition of the variable; but at least the first call should work.

At any rate, to continue the analysis of `tf.function` we have 3 options:

1. Declare `f` as a function that accepts an input parameter: the parameter can be a `tf.Variable` or any other input type.
2. Create a function that inherits the Python variable from the parent scope, and check in the function body if it has already been declared (`if b != None`).
3. Wrap everything inside a class. The `__call__` method is the function we want to execute and the variable is declared as a private attribute (`self._b`). The same declaration check of the point 2 has to be performed. In practice this is the Object Oriented solution that is functionally equivalent to the one suggested in the point 2.

In order to understand if there are differences among these methods, all of them are going to be analyzed. Since points 2 and 3 are equivalent the analysis takes in consideration only the point 1 and 2 as possible scenarios.

## tf.function: handling states using input parameters

In order to workaround the problem of the variable declaration inside the function body, we can refactor the `f` function making it accept `b` as input parameter.

It should be pretty clear that `tf.function` do not allow to simply wrap a function that works in eager mode and accelerate it - it requires to think about how the conversion is performed, what happens when converting Python to graph operations, and take care of **a lot of subtleties**.

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

What happens? Points 1,2, and 3 the same of the previous analysis. But now we have a different point 4 since there is no failure.

The evaluation of `f(b)` produces the following output

```
PRINT:  Tensor("add:0", shape=(2, 2), dtype=float32)
TF-PRINT:  [[22 22]
 [23 13]]
```

The first `PRINT` is the Tensor, Tensorflow 1.x style that is used to build the computational graph.
The second line, `TF-PRINT` is the result of the session evaluation of the graph defined.

Please note that when the Python function is traced and the graph is being built the `tf.Tensor` objects have a different semantic.
When outside of a `tf.function` decorator they are tensor that hold a value. When the decoration is present no value stored and they just represent a description of the computation, as it happens in Tensorflow 1.x.

What happens if the function is invoked more than once?

```python
f(b)
f(b)
```

It simply produces:

```
TF-PRINT:  [[22 22]
 [23 13]]
TF-PRINT:  [[22 22]
 [23 13]]
```

The Python print statement has been invoked only when the graph has been built - we are executing only the graph without redefining it.

What happens if instead of a `tf.Variable` we pass a `tf.Tensor` or a Python type?

## tf.function: changing the input type
