---
layout: post
title: "Analyzing tf.function to discover AutoGraph strengths and subtleties - part 1"
date: 2019-03-21 08:00:00
categories: tensorflow tf.function
summary: "AutoGraph is one of the most exciting new features of Tensorflow 2.0: it allows transforming a subset of Python syntax into its portable, high-performance and language agnostic graph representation bridging the gap between Tensorflow 1.x and the 2.0 release based on eager execution. As often happens all that glitters is not gold: although powerful, AutoGraph hides some subtlety that is worth knowing; this article will guide you through them using an error-driven approach."
---

AutoGraph is one of the most exciting new features of Tensorflow 2.0: it allows transforming a subset of Python syntax into its portable, high-performance and language agnostic graph representation bridging the gap between Tensorflow 1.x and the 2.0 release based on eager execution.

As often happens all that glitters is not gold: although powerful, AutoGraph hides some subtlety that is worth knowing; this article will guide you through them using an error-driven approach.

## Session execution

The reader familiar with Tensorflow 1.x already knows that the standard workflow to get a callable graph (or better, define a graph with nodes that can be executed within a `tf.Session`) is:

1. Create the `tf.Graph` object and set it as the default graph for the current scope.
2. Describe the computation using the Tensorflow API (e.g. `y = tf.matmul(a,x) + b`).
3. Think in advance about variable sharing and define the variables scope accordingly.
4. Create and configure the `tf.Session`.
5. Build the concrete graph and load it into the `tf.Session`.
6. Initialize all the variables.
7. Use the `tf.Session.run` method to start the computation. The node execution will trigger a backtracking procedure from the chosen nodes (`.run` input parameters) to their inputs, in order to resolve the dependencies and compute the result.

All these points can be translated in code with this minimal example:

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
    print(sess.run(y))
```

Tensorflow 2.0, defaulting on eager execution follows a completely different approach based on the direct execution of what the user wants.

- Remove the graph definition.
- Remove the session execution.
- Remove variables initialization.
- Remove the variable sharing via scopes.
- Remove the `tf.control_dependencies` to execute sequential operation not connected by a dependency relation.

Just write the code and run it:

```python
a = tf.constant([[10,10],[11.,1.]])
x = tf.constant([[1.,0.],[0.,1.]])
b = tf.Variable(12.)
y = tf.matmul(a, x) + b
print(y.numpy())
```

The eager counterpart of any Tensorflow 1.x source code is usually slower since it relies on the Python interpreter to run the computation and there are a lot of optimizations that are only possible on DataFlow graphs.

The bridge among the two versions that allow creating computational graphs even in Tensorflow 2.0 is `tf.function`.

## tf.function, not tf.Session

One of the major changes in Tensorflow 2.0 is the removal of the `tf.Session` object (see [RFC: Functions, not Sessions](https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md)). This change forces the user to organize the code in a better way: no more a `tf.Session` object to pass around, but just Python functions that can be accelerated with a simple decoration.

In order to define a graph in Tensorflow 2.0, we need to define a Python function and decorate it with `@tf.function`.

> **Note**: the speed-up is not guaranteed. There are certain tasks in which is not worth converting the function to its graph representation, as is the case of this simple matrix multiplication we are performing here.
> However, for computationally intensive tasks like the optimization of a deep neural network the Graph conversion provides a huge performance boost.

The automatic conversion from Python code to its graph representation is called AutoGraph.

In Tensorflow 2.0, AutoGraph is automatically applied to a function when it is decorated with [@tf.function](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/function); this decorator creates callable graphs from Python functions.

### tf.function: layman explanation

On the first call of a `tf.function` decorated function:

- The function is executed and traced. Eager execution is disabled in this context, therefore every `tf.` method just define a `tf.Operation` node that produces a `tf.Tensor` output, Tensorflow 1.x like.
- AutoGraph is used to detect Python constructs that can be converted to their graph equivalent (`while` → `tf.while`, `for` → `tf.while`, `if` → `tf.cond`, `assert` → `tf.assert`, ...).
- From the function trace + autograph, the graph representation is built. In order to preserve the execution order in the defined graph, `tf.control_dependencies` is automatically added after every statement, in order to condition the line $$i+1$$ on the execution of line $$i$$.
- The `tf.Graph` object has now been built.
- Based on the function name and the input parameters a unique ID is created and associated with the graph. The graph is cached into a map: `map[id] = graph`.
- Any function call will just re-use the defined graph if the key matches.

The next sections will guide you through the required steps to migrate a 1.x snippet to its eager and graph-accelerated version.

## Conversion to eager execution

To use `tf.function` the first thing to do is to refactor the old 1.x code, wrapping the code we want to execute into a session.

In general, where first there was a session execution, now there is Python function.

**Note**: this is a huge advantage since the software architecture it allows defining is cleaner, and easy to maintain and document.

```python
def f():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    return y
```

What happens now? **Nothing**. Tensorflow 2.0 works in eager mode by default, this means that we just defined a standard Python function and if we evaluate it:

```python
print(f().numpy())
```

We get the expected result:
```
[[22. 22.]
 [23. 13.]]
```

## From eager to tf.function: the need to refactor

Let's just add the `@tf.function` decoration to the `f` function. For the sake of clarity (and to debug in the old-school print driven way) let's add even a `print` and a `tf.print` statement inside the function body:

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

1. The annotation `@tf.function` wrapped the `f` function in a `tensorflow.python.eager.def_function.Function` object. The Python function is assigned to the `.python_function` property of the object.
2. Until the object is called ( `f()` ): nothing happens.
3. When `f()` is called the process of graph building starts. At this stage, only the Python code is executed and the behavior of the function is traced, in order to collect the required data to build the graph. Thus the only output we get is:
```
PRINT:  Tensor("add:0", shape=(2, 2), dtype=float32)
```
The `tf.print` call is not evaluated as any other `tf.*` method, since Tensorflow already knows everything about that statements and it can use them as they are to build the graph.
4. **FAIL**: during the first and only invocation of the function, the following exception has been raised
```
ValueError: tf.function-decorated function tried to create variables on non-first call.
```
`@tf.function` failed to build the graph.

I thought I had found a bug so I opened [an issue](https://github.com/tensorflow/tensorflow/issues/26812). The [RFC: Functions, not Session](https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md#functions-that-create-state) in the section dedicated to the functions that create a state clearly states:

> State (like tf.Variable objects) are only created the first time the function f is called.

Therefore I expected an execution flow like:

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

But in practice, as [Alexandre Passos](https://github.com/tensorflow/tensorflow/issues/26812#issuecomment-474595919) pointed out, this can happen because there is no guarantee about the number of times `tf.function` evaluates the Python function while converting it to Graph. Therefore the behavior described above is exactly what happens under the hood.

However, it still remains shady when this second function call is performed and why there is no a second output from the `print` call (that it should be executed since is before the `tf.Variable` definition).

As it's easy to understand, the exception is raised because the function contains a `tf.Variable` definition. In fact, a `tf.Variable` in eager mode is just a plain Python object, that gets destroyed as soon as it goes out of scope. While a `tf.Variable` object defines a persistent object if the function is decorated: in fact, the eager mode is disabled and the `tf.Variable` object defines a node in a persistent Graph (a Graph that exists even after the session execution).

Hence, the same function that in eager mode is perfectly valid (and in fact the same function without annotation works), when annotated with `@tf.function` stops working. Thus this is the **first lesson**:

> Converting a function that works in eager mode to its Graph representation requires to think about the Graph even though we are working in eager mode.

So now, what we have to do in order to go on with the analysis of `tf.function`? There are 3 options:

1. Declare `f` as a function that accepts an input parameter: the parameter can be a `tf.Variable` or any other input type.
2. Create a function that inherits the Python variable from the parent scope, and check in the function body if it has already been declared (`if b != None`).
3. Wrap everything inside a class. The `__call__` method is the function we want to execute and the variable is declared as a private attribute (`self._b`). The same declaration check of point 2 has to be performed. In practice, this is the Object Oriented solution that is functionally equivalent to the one suggested in point 2.

In order to understand if there are differences among these methods, all of them are going to be analyzed.

## Handling states breaking the function scope

Points 2 and 3 described above have the same behavior, but the Object Oriented solution is way better from the software engineering point of view. Just compare these two implementations:

**The ugly solution with global variables** (highly discouraged):

```python
b = None

@tf.function
def f():
    a = tf.constant([[10, 10], [11., 1.]])
    x = tf.constant([[1., 0.], [0., 1.]])
    global b
    if b is None:
        b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y

f()
```

**Object Oriented solution** (recommended):

```python
class F():
    def __init__(self):
        self._b = None

    @tf.function
    def __call__(self):
        a = tf.constant([[10, 10], [11., 1.]])
        x = tf.constant([[1., 0.], [0., 1.]])
        if self._b is None:
            self._b = tf.Variable(12.)
        y = tf.matmul(a, x) + self._b
        print("PRINT: ", y)
        tf.print("TF-PRINT: ", y)
        return y

f = F()
f()
```

The Object Oriented solution is superior: no global variables, the class `F` can always be instantiated and called without having to worry about the global `b` variable that every other function sees.

So far so good, we solved the problem of functions that create states by breaking the scope. In fact, once executed the previous script returns the same values of the eager execution.

From this the **second lesson**:

> When defining a function you want to accelerate converting it to its graph representation, you have to define its body thinking about the Graph is being built.
> There is no 1:1 match between eager execution and the graph built by `@tf.function`; thanks to AutoGraph there is no need to worry about the order of the operation execution, but special attention is required when definition function with objects that can create a state (`tf.Variable`).

A second option to solve the problem is to move the variable outside the function body.

## Handling states using input parameters

We can refactor the `f` function making it accept `b` as an input parameter.

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

As in the previous section, the function produces the expected behavior. Moreover, being passed by reference the status (the variable value) can be updated from inside the graph accelerated function while being available from the outside. In fact, the following code produces 1,2,3.

```python
a = tf.Variable(0)

@tf.function
def g(x):
    x.assign_add(1)
    return x

print(g(a))
print(g(a))
print(g(a))
```

## Conclusions

This is the end of part 1. The article is divided into 2 parts because there are a lot of things to write about `tf.function` and its subtleties and a single article is going to be too long.

In part 1 we learned how to convert a 1.x code to its eager version, how to convert the eager version to its graph representation concluding with the problems to face when working with functions that create a state.

In the next part, we'll study what happens when instead of a `tf.Variable` we pass a `tf.Tensor` or a Python value as input to a decorated function, together with the analysis of the `tf.function` behavior when the Python code is executed in the first function call: are we sure everything is going to be converted to the Graph representation we expect?

Stay tuned for part 2!

If you find this article useful, feel free to share it using the buttons below!


