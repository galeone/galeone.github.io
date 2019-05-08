---
layout: post
title: "Analyzing tf.function to discover AutoGraph strengths and subtleties - part 3"
date: 2019-04-28 08:00:00
categories: tensorflow tf.function
summary:TODO
---

In [part 1](/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/) we learned how to convert a 1.x code to its eager version, the eager version to its graph representation and faced the problems that arise when working with functions that create a state.

In [part 2](/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/) we learned that `tf.function` creates a new graph for every different input value, if the input type is not a `tf.Tensor` object and that this could slow down (or speed up if correctly used) the execution. Moreover, the differences between the `tf.autograph` generated source code and what happens, instead, when using AutoGraph trough `tf.function` have been highlighted.

In this third and last part, we'll analyze what happens when `tf.function` is used to convert a function that contains complex Python constructs in its body and how different graphs can or can't interact each other. Should we design functions thinking about how they are going to be converted?

## AutoGraph capabilities and limitations

In the Tensorflow repository, in the `python/autograph` folder, we can find [a document](https://github.com/tensorflow/tensorflow/blob/560e2575ecad30bedff5b192f33f6d06b19ccaeb/tensorflow/python/autograph/LIMITATIONS.md) that explains which are the capabilities and the limitations of the AutoGraph module together with a list of the Python constructs it is able to convert.

The [table](https://github.com/tensorflow/tensorflow/blob/560e2575ecad30bedff5b192f33f6d06b19ccaeb/tensorflow/python/autograph/LIMITATIONS.md#python-language-support-status) in the section "Python Language Support Status" contains a list of all the Python constructs that AutoGraph explicitly supports, plan to support, or won't support. Among them, we can find the widely used `while`, `for`, `if`  statements, the Python built-in `print`, `len`, `range`, and the iterator construct.

In the next sections various Python functions that uses these Python constructs are analyzed, in order to find if the function body gets converted as we expect or it is required to design the functions thinking about the graph conversion.

## if ... else

Here's the function we are going to analyze:

```python
@tf.function
def if_else(a, b):
  if a > b:
    tf.print("a > b", a, b)
  else:
    tf.print("a <= b", a, b)
```

It's trivial: when `a` is greater than `b` then it prints `a > b` followed by the value of `a` and `b`; otherwise it prints `a <= b` and their value.

**Step 1: graph conversion**

As seen in the previous articles, the `tf.autograph` package can be used to inspect the result of the graph conversion.

```python
print(tf.autograph.to_code(if_else.python_function))
```

The generated code is:

```python
def tf__if_else(a, b):
    cond = a > b

    def get_state():
        return ()

    def set_state(_):
        pass

    def if_true():
        ag__.converted_call(
            "print",
            tf,
            ag__.ConversionOptions(
                recursive=True,
                force_conversion=False,
                optional_features=(),
                internal_convert_user_code=True,
            ),
            ("a > b", a, b),
            None,
        )
        return ag__.match_staging_level(1, cond)

    def if_false():
        ag__.converted_call(
            "print",
            tf,
            ag__.ConversionOptions(
                recursive=True,
                force_conversion=False,
                optional_features=(),
                internal_convert_user_code=True,
            ),
            ("a <= b", a, b),
            None,
        )
        return ag__.match_staging_level(1, cond)

    ag__.if_stmt(cond, if_true, if_false, get_state, set_state)
```

The conversion is trivial too: the `if_stmt` maps, more or less, with the [`tf.cond`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/cond) function; the first parameter is the condition to check, the second is the branch to take when the condition is `True`, the third the branch to take otherwise.
The `get_state` and `set_state` methods basically do nothing and we can safely ignore them.

**Step 2: execution**

As seen in [part 2](/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/) 'tf.function` by design does not do the boxing of the Python native types; therefore we use a `tf.Tensor` produced by a `tf.constant` operation as input.

```python
x = tf.constant(1)
if_else(x, x)
```

As expected, the output is: `a <= b 1 1`.

## if ... elsif ... else

Let's change the function a little bit, adding an `elif` statement. The function now is:

```python
@tf.function
def if_elif(a, b):
  if a > b:
    tf.print("a > b", a, b)
  elif a == b:
    tf.print("a == b", a, b)
  else:
    tf.print("a < b", a, b)
```

**Step 1: graph conversion**

The generated function, with the removed `tf.print` conversion and `(get|set)\_state` function definitions,is

```python
def tf__if_elif(a, b):
    cond_1 = a > b

    def if_true_1():
        # tf.print("a > b", a, b)
        return ag__.match_staging_level(1, cond_1)

    def if_false_1():
        cond = a == b

        def if_true():
            # tf.print(a == b, a, b)
            return ag__.match_staging_level(1, cond)

        def if_false():
            # tf.print(a < b, a,b)
            return ag__.match_staging_level(1, cond)

        ag__.if_stmt(cond, if_true, if_false, get_state, set_state)
        return ag__.match_staging_level(1, cond_1)

    ag__.if_stmt(cond_1, if_true_1, if_false_1, get_state_1, set_state_1)
```

The conversion seems correct: two `tf.cond` nested. The inner `tf.cond` is defined inside the false branch of the outer one. The outer `tf.cond` checks if `a > b`, and if it is `True` then it prints `a > b`, otherwise executes the `if_false_1` branch that contains the inner `tf.cond`.

The inner `tf.cond` has the equality condition `cond = a == b` to verify; if it holds, it prints 'a == b`, otherwise it prints `a < b`.


**Step 2: execution**

```python
x = tf.constant(1)
if_elif(x, x)
```

Executing it we expect to see `a == b, 1, 1`  since this is the truth. However, the output is `a < b 1 1`. **WHAT!?**

Ok then, let's debug.

**Step 3: debugging**

The AutoGraph representation looks correct, moreover, we can try by removing the `tf.function` annotation to see if everything goes as expected in eager mode.

```python
def if_elif_eager(a, b):
  if a > b:
    tf.print("a > b", a, b)
  elif a == b:
    tf.print("a == b", a, b)
  else:
    tf.print("a < b", a, b)
x = tf.constant(1)
if_elif_eager(x, x)
```

The output is correct `a == b 1 1`, that's exactly what this simple snippet should do!

However, let's see if some weird behavior happens even when using the standard eager execution.


## for ... in range

## while

## Interacting Graphs


## Conclusions
