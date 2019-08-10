---
layout: post
title: "Analyzing tf.function to discover AutoGraph strengths and subtleties - part 2"
date: 2019-04-03 08:00:00
categories: tensorflow tf.function
summary: "In part 1 we learned how to convert a 1.x code to its eager version, the eager version to its graph representation and faced the problems that arise when working with functions that create a state. In this second part, we’ll analyze what happens when instead of a tf.Variable we pass a tf.Tensor or a Python native type as input to a tf.function decorated function. Are we sure everything is going to be converted to the Graph representation we expect?"
authors:
    - pgaleone
---

In [part 1](/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/) we learned how to convert a 1.x code to its eager version, the eager version to its graph representation and faced the problems that arise when working with functions that create a state.

In this second part, we’ll analyze what happens when instead of a `tf.Variable` we pass a `tf.Tensor` or a Python native type as input to a `tf.function` decorated function. Are we sure everything is going to be converted to the Graph representation we expect?

## tf.function uses AutoGraph

For sake of clarity, below is reported the complete signature of the `tf.function` function:

```python
def function(func=None,
             input_signature=None,
             autograph=True,
             experimental_autograph_options=None)
```

The default value of the `autograph` parameter is `True`, hence this means that `tf.function` uses AutoGraph. The documentation describes what happens when `autograph` is `True` or `False`.

Quoting the [documentation](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/function):

- When `autograph` is `True`, all Python code that depends on Tensor values is staged into a TensorFlow graph.
- When `autograph` is `False`, the function is traced and control flow is not allowed to depend on data.

Thus, by default `tf.function` uses AutoGraph and we are going to analyze how it behaves by changing the input types and the function structure.

## Changing tf.Tensor input type

Let's start by defining our test Python function. The function parameters type is of fundamental importance since is used to create a graph, that is a statically typed object, and to assign it an ID (for a complete and informal explanation of what's going on when calling the function for the first time, see [tf.function: layman explanation](/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/#tffunction-layman-explanation)).

```python
@tf.function
def f(x):
    print("Python execution: ", x)
    tf.print("Graph execution: ", x)
    return x
```

Here's the brief function description:

- **Line 1**: the function accepts a Python variable `x` that can be literally everything.
- **Line 2**: the `print` function is executed once, only during the function creation.
- **Line 3**: the `tf.print` function is executed every time the graph is evaluated.
- **Line 4**: `x` is returned.

Let's see if everything goes as we expect by running some test.

```python
print("##### float32 test #####")
a = tf.constant(1, dtype=tf.float32)
print("first call")
f(a)
a = tf.constant(1.1, dtype=tf.float32)
print("second call")
f(a)

print("##### uint8 test #####")

b = tf.constant(2, dtype=tf.uint8)
print("first call")
f(b)
b = tf.constant(3, dtype=tf.uint8)
print("second call")
f(b)
```

Everything goes as we expect:

```
##### float32 test #####
first call
Python execution:  Tensor("x:0", shape=(), dtype=float32)
Graph execution:  1
second call
Graph execution:  1.1
##### uint8 test #####
first call
Python execution:  Tensor("x:0", shape=(), dtype=uint8)
Graph execution:  2
second call
Graph execution:  3
```

A graph is created for every different input type of the `tf.Tensor` object passed. We can have a look at the graph version of the function `f` by using the `tf.autograph` module.

```python
tf.autograph.to_code(f.python_function)
```
returns a string that represents the graph code of the python function `f`.

```python
def tf__f(x):
  try:
    with ag__.function_scope('f'):
      do_return = False
      retval_ = None
      with ag__.utils.control_dependency_on_returns(ag__.converted_call(print, None, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=ag__.Feature.ALL, internal_convert_user_code=True), ('Python execution: ', x), {})):
        tf_1, x_1 = ag__.utils.alias_tensors(tf, x)
        with ag__.utils.control_dependency_on_returns(ag__.converted_call('print', tf_1, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=ag__.Feature.ALL, internal_convert_user_code=True), ('Graph execution: ', x_1), {})):
          x_2 = ag__.utils.alias_tensors(x_1)
          do_return = True
          retval_ = x_1
          return retval_
  except:
    ag__.rewrite_graph_construction_error(ag_source_map__)
```

The code is machine generated and therefore hard to read. However, we can notice something interesting: in the Graph code, we can still find some reference to the Python code that's executed only on the first-call.

(rewritten and formatted for clarity)
```python
with ag__.utils.control_dependency_on_returns(
        ag__.converted_call(
            print, None, ag__.ConversionOptions(
                recursive=True,
                force_conversion=False,
                optional_features=ag__.Feature.ALL,
                internal_convert_user_code=True),
            ('Python execution: ', x), {})
        ):
```

We can see that `ag__.utils.control_dependency_on_returns` creates a `tf.control_dependency` context manager when the function created by `converted_call` returns. This allows preserving the execution order in graph mode, forcing the execution of the nodes to be sequential.

The [`converted_call`](https://github.com/tensorflow/tensorflow/blob/56c8527fa73f694b76963dbb28a9d011d233086f/tensorflow/python/autograph/impl/api.py#L206) function compiles a function call. This function has all the information required to convert and execute a function call (`print` in this case) as we can see analyzing its signature. That is `(f, owner, options, args, kwargs)` where:

- `f` is the function we are interested in calling. In this case `print`, in the next call (graph execution) the string `'print'`.
- `owner` is the function package or its owner. In this case is `None` since `print` is the standard Python print function, in the next call instead is `tf_1` that is an alias for the `tf` package.
- `options` are the conversion options.
- `args` are the function `f` (`print`) positional arguments.
- `kwargs` are the function `f` (`print`) named arguments.

**Question Time**:

Begin a graph, why is there a reference to the Python code that it should be executed only on the first call when tracing the function execution?

**Hypothesis**:

My guess is that while tracing the function execution on the first-call there is no easy way to understand if a Python function produces some side effect that will change the graph behavior and for this reason (and to be sure to preserve the execution order) every python function invoked on the first-call is traced and added to the graph.

If during the execution of the function (in this case a print, but it could be any arbitrary complex function) a side effect is detected (a call to a `tf.` method) then the graph code is updated; otherwise, as in this case, the operation is converted to a `tf.no_op` by the `converted_call`.

This is my guess since we don't see any output or side effect generated by the `print` converted call, while we do see them when using the `tf.print` function (now, node).

Since this behavior is not clear, and I'm only guessing what's going on, [I asked on the Tensorflow Developers Group](https://groups.google.com/a/tensorflow.org/forum/#!topic/developers/SD_ijT4MuPw) an explanation; if you are curious about this behavior please subscribe to the mailing list and monitor that thread!

## Using a Python native type

The function input is not constrained to be a `tf.Tensor` object; AutoGraph understands the input type and creates a new Graph accordingly. Let's see if there is something different between using a `tf.Tensor` object (that has a `dtype` parameter and thus has a well-defined type) and a Python native type.

Since in Python there are only three distinct numeric types (integers, floating point numbers, and complex numbers) we'll test how `tf.function` behaves with all of them.

```python
def printinfo(x):
  print("Type: ", type(x), " value: ", x)

print("##### int test #####")
print("first call")
a = 1
printinfo(a)
f(a)
print("second call")
b = 2
printinfo(b)
f(b)

print("##### float test #####")
print("first call")
a = 1.0
printinfo(a)
f(a)
print("second call")
b = 2.0
printinfo(b)
f(b)

print("##### complex test #####")
print("first call")
a = complex(1.0, 2.0)
printinfo(a)
f(a)
print("second call")
b = complex(2.0, 1.0)
printinfo(b)
f(b)
```

Differently from the `tf.Tensor` input case, the behavior is different from the expected:

```
##### int test #####
first call
Type:  <class 'int'>  value:  1
Python execution:  1
Graph execution:  1

second call
Type:  <class 'int'>  value:  2
Python execution:  2
Graph execution:  2

##### float test #####
first call
Type:  <class 'float'>  value:  1.0
Graph execution:  1
second call
Type:  <class 'float'>  value:  2.0
Graph execution:  2

##### complex test #####
first call
Type:  <class 'complex'>  value:  (1+2j)
Python execution:  (1+2j)
Graph execution:  (1+2j)
second call
Type:  <class 'complex'>  value:  (2+1j)
Python execution:  (2+1j)
Graph execution:  (2+1j)
```

We expect a Graph for type, but instead, if we look carefully **we have a different graph per input value**. In fact:

1. The first call `f(1)` executes the Python code, traces its execution, creates a Graph and executes it.
2. The second call `f(2)` executes the Python code **again**, traces its execution, creates a Graph and executes it.
3. The first call `f(1.0) **does not execute the python code**, therefore is using an already existing Graph.
4. The second call `f(2.0)` **does not execute the python code**, therefore behaves like 3.
5. The first call `f(1 + 2j)` behaves like 1.
6. The first call `f(2 + 1j)` behaves like 2.

This is weird.

Are we sure that `tf.function` is creating a different Graph for every **input value**? Let's check it.

The function `f` returns the input parameter `x` passed as input, thus we can check if the input type returned matches the one passed. Since we defined first the graph for the integer values 1 and 2, let's see if the call `f(1.0)` and `f(2.0)` return, as we expect, the integer value or not.

```python
ret = f(1.0)
if tf.float32 == ret.dtype:
    print("f(1.0) returns float")
else:
    print("f(1.0) return ", ret)
```

The output is

```
Graph execution:  1
f(1.0) return  tf.Tensor(1, shape=(), dtype=int32)
```

We can conclude that the ID associated with the graph is built using the input parameters **value** when using Python native types (`1.0 == 1`) causing this weird behavior.

**Warning**: this is highly inefficient since every time a `tf.function` decorated function is called with a different input value, both the Python execution + tracing and Graph creation must be executed, making the Graph conversion useless.

### Performance measurement

The following code is a simple benchmark to check if the previous reasoning is correct.

```python
@tf.function
def g(x):
  return x

start = time.time()
for i in tf.range(1000):
  g(i)
end = time.time()

print("tf.Tensor time elapsed: ", (end-start))

start = time.time()
for i in range(1000):
  g(i)
end = time.time()

print("Native type time elapsed: ", (end-start))
```

The `g` function, decorated with `tf.function`, is executed the first time in a loop of `tf.Tensor` objects, all created with the same dtype `tf.int32` by the `tf.range` call, while the second time it is executed in a loop of Python integers.

The benchmark confirms the hypothesis:

```
tf.Tensor time elapsed:  0.41594886779785156
Native type time elapsed:  5.189513444900513
```

Conclusion: **do use `tf.Tensor` everywhere**.

AutoGraph is highly optimized and works well when the input is a `tf.Tensor` object, while it creates a **new graph** for every different input parameter value with a **huge drop in performance**.

## Is tf.function really using AutoGraph?

The signature of `tf.function` clearly states that `autograph=True`, therefore we expect that the function is converted by using tracing + autograph; let's check if it is true.

Experimenting I found out that there are functions (like functions without a return value) that `tf.autograph.to_code` can't convert, but that work correctly if decorated with `tf.function` and called.

Let's just update the function `f` removing the `return` statement.

```python
@tf.function
def f(x):
    print("Python execution: ", x)
    tf.print("Graph execution: ", x)
```

If called twice it works correctly (`f(1)` followed by `f(1)`): it executes the `print` function only on the first call.
On every non-first call, it prints only the `tf.print` output.

This is the behavior we expect when a function is traced first, then graph converted, then an ID is assigned to the graph and the graph is being re-used.

Something weird happens if we try to look at the code, as we did it before, generated by the autograph `to_code` method; in fact

```python
tf.autograph.to_code(f.python_function)
```

raises the following exception:

```
ValueError during conversion: Unable to insert statement into the computation flow: it is not followed by any computation which the statement could gate.
```

**Question time**

So the question is: shouln't `tf.function` use `tf.autograph.to_code`/`tf.autograph.to_graph` under the hood? How is the graph being built if `to_code` fails?


**Hypothesis**:

It seems that `tf.function` builds and stores a graph correctly since on every non-first call we got only the `tf.print` output.
However, `tf.autograph.to_code` fails raising a `ValueError`; if `tf.function` uses `tf.autograph` internally, probably this exception is caught inside the `tf.function` body and handled somehow in order to simulate the graph execution.

This is just a guess and in the previously linked [thread](https://groups.google.com/a/tensorflow.org/forum/#!topic/developers/SD_ijT4MuPw), I've asked for an additional explanation of this behavior; as for the previous question, if you are curious about this behavior please subscribe to the mailing list and monitor that thread!


## Conclusions

In this article we analyzed the behavior of `tf.function` when AutoGraph is used and discovered that a `@tf.function` decorated function:

- behaves as we expect if we use a `tf.Tensor` as input type (please use `tf.Tensor` everywhere!),
- creates a new graph (or it behaves like is creating it) if the input is a Python native type; in this case, the graph seems created by looking at the parameters' value (please use `tf.Tensor` everywhere!)²,
- reuses the first graph created for a Python native type comparing the parameters input values (not the type, thus `1.0 == 1`),
- contains still a reference to the Python code executed only once in the graph-converted code.

In addition, we found out that certain functions work with `tf.function` but the autograph module can't convert them; how is this possible?

This second part was thought to be the last one, but due to the overwhelming list of peculiarities to know and analyze, the behavior of `tf.function` and AutoGraph when the function body is more complex than a `print` function invocation haven't been analyzed yet; so stay tuned for the part 3!

If you find this article useful, feel free to share it using the buttons below!
