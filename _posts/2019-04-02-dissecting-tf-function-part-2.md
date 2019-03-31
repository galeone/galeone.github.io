---
layout: post
title: "Analyzing tf.function to discover AutoGraph strengths and subtleties - part 2"
date: 2019-03-21 08:00:00
categories: tensorflow tf.function
summary: ""
---

In [part 1](/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/) we learned how to convert a 1.x code to its eager version, the eager version to its graph representation and faced the problems that arise when working with functions that create a state.

In this second part, we’ll study what happens when instead of a `tf.Variable` we pass a `tf.Tensor` or a Python variable with a different type as input to a `tf.function` decorated function, together with the analysis of the AutoGraph behavior when the Python code is executed in the first function call: are we sure everything is going to be converted to the Graph representation we expect?

## Changing tf.Tensor input type

Let's start by defining a Python function. The function parameters type is of fundamental importance since is used to create a graph, that is a statically typed object, and to assigning it an ID (for a complete and informal explanation of what's going on when calling the function for the first time, see [tf.function: layman explanation](/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/#tffunction-layman-explanation)).

```python
@tf.function
def f(x):
    print("Python execution: ", x)
    tf.print("Graph execution: ", x)
    return x
```

Brief function description:

- **Line 1**: the function accepts a Python variable `x` that can be literally everything.
- **Line 2**: the first `print` statement is executed once, only during the function creation.
- **Line 3**: the second `tf.print` statement is executed every time the graph is evaluated.
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

The code is machine generated and therefore hard to read. However we can notice something interesting: in the Graph code we can still find some reference to the Python code that's executed only on the first-call.

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

We can see that `ag__.utils.control_dependency_on_returns` creates a `tf.control_dependency` context manager when the function created by `converted_call` returns. This allow preserving the execution order in graph mode, forcing the execution of the nodes to be sequential.

The [`converted_call`](https://github.com/tensorflow/tensorflow/blob/56c8527fa73f694b76963dbb28a9d011d233086f/tensorflow/python/autograph/impl/api.py#L206) function compiles a function call. This function has all the information required to convert and execute a function call (`print` in this case) as we can see analyzing its signature. That is `(f, owner, options, args, kwargs)` where:

- `f` is the function we are interested in calling. In this case `print`, in the next call (graph execution) the string `'print'`.
- `owner` is the function package or its owner. In this case is `None` since `print` is the standard Python print function, in the next call instead is `tf_1` that is an alias for the `tf` package.
- `options` are the conversion options.
- `args` are the function `f` (`print`) positional arguments.
- `kwargs` are the function `f` (`print`) named arguments.

#### Question

Begin a graph, why is there a reference to the Python code that is executed only on the first call?

#### Guess

My guess is that while tracing the function execution on the first-call there is no easy way to understand if a Python function produces some side effect that will change the graph behavior and for this reason (and to be sure to preserve the execution order) every python function invoked on the first-call is traced and added to the graph.

If during the execution of the function (in this case a print, but it could be any arbitrary complex function) a side effect is detected (a call to a `tf.` method) then the graph code is updated; otherwise, as in this case, the operation is converted to a `tf.no_op` by the `converted_call`.

This is my guess since we don't see any output or side effect generated by the `print` converted call, while we do see them when using the `tf.print` function (now, node).

### Changing Python type

Instead of passing a `tf.Tensor` object, AutoGraph and `tf.function` allow us to pass a Python variable.


## Conclusions

If you find this article useful, feel free to share it using the buttons below!
