---
layout: post
title: "Analyzing tf.function to discover AutoGraph strengths and subtleties - part 3"
date: 2019-05-10 12:00:00
categories: tensorflow tf.function
summary: "In this third and last part, we analyze what happens when tf.function is used to convert a function that contains complex Python constructs in its body. Should we design functions thinking about how they are going to be converted?"
authors:
    - pgaleone
---

In [part 1](/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/) we learned how to convert a TensorFlow 1.x code to its eager version, the eager version to its graph representation, and faced the problems that arise when working with functions that create a state.

In [part 2](/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/) we learned that `tf.function` creates a new graph for every different input value if the input is not a `tf.Tensor` object but a Python native type and how this could slow down (or speed up if correctly used) the execution. Moreover, the differences between the `tf.autograph` generated source code and what happens, instead, when using AutoGraph trough `tf.function` have been highlighted.

In this third and last part, we analyze what happens when `tf.function` is used to convert a function that contains "complex" Python constructs in its body. Should we design functions thinking about how they are going to be converted?

## AutoGraph capabilities and limitations

In the TensorFlow repository, in the `python/autograph` folder, we can find [a document](https://github.com/tensorflow/tensorflow/blob/560e2575ecad30bedff5b192f33f6d06b19ccaeb/tensorflow/python/autograph/LIMITATIONS.md) that explains which are the capabilities and the limitations of the AutoGraph module together with a list of the Python constructs it is able to convert.

The [table](https://github.com/tensorflow/tensorflow/blob/560e2575ecad30bedff5b192f33f6d06b19ccaeb/tensorflow/python/autograph/LIMITATIONS.md#python-language-support-status) in the section "Python Language Support Status" contains all the Python constructs that AutoGraph explicitly supports, plan to support, or won't support. Among them, we can find the widely used `while`, `for`, `if`  statements, the Python built-in `print`, `len`, `range`, and the iterator construct.

In the next sections, various Python functions that use these Python constructs are analyzed, to understand if the function body gets converted as we expect or if it is required to design the functions thinking about the graph conversion.

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

As seen in [part 2](/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/) `tf.function` by design does not do the boxing of the Python native types; therefore we use a `tf.Tensor` produced by a `tf.constant` operation as input.

```python
x = tf.constant(1)
if_else(x, x)
```

As expected, the output is: `a <= b 1 1`.

## if ... elif ... else

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

The generated function, with the removed `tf.print` conversion and `(get|set)\_state` function definitions, is

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

The inner `tf.cond` has the equality condition `cond = a == b` to verify; if it holds, it prints `a == b`, otherwise it prints `a < b`.

**Step 2: execution**

```python
x = tf.constant(1)
if_elif(x, x)
```

Executing it, we expect to see `a == b, 1, 1`  since this is the truth. However, the output is `a < b 1 1`. **WHAT?**

OK then, debug time.

<hr />

*Update (14 Sept 2019): as [Raphael Meudec](https://twitter.com/raphaelmeudec) pointed out in the tweet below, this behavior has been changed in TensorFlow 2.0-rc0 and it works as expected.
However, the lessons presented later in the article are still valid: following them helps you writing idiomatic TensorFlow 2.0 code.*

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hey <a href="https://twitter.com/paolo_galeone?ref_src=twsrc%5Etfw">@paolo_galeone</a>, great blog post series on tf.function! I&#39;ve tried the if_elif_else case (from part 3: <a href="https://t.co/HukmaUY4dL">https://t.co/HukmaUY4dL</a>) this afternoon, and it looks like it has been fixed in 2.0.0rc0. Thought you might want to know</p>&mdash; Raphael Meudec (@raphaelmeudec) <a href="https://twitter.com/raphaelmeudec/status/1172510929659019264?ref_src=twsrc%5Etfw">September 13, 2019</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<hr />

**Step 3: debugging**

The AutoGraph representation looks correct. Moreover, we can try by using the non-converted function to see if everything goes as expected in eager mode.

```python
x = tf.constant(1)
if_elif.python_function(x, x)
```

In eager mode the output is correct: `a == b 1 1`. So we do expect to see the same output when we feed the function with two `tf.Tensor` objects that hold the same value

```python
x, y = tf.constant(1), tf.constant(1)
if_elif.python_function(x, y)
```

Surprise! The output is `a < b 1 1`. *What's going on?*

### Lesson 1: not all operators are created equal

This lesson is not about AutoGraph or `tf.function` but is about `tf.Tensor`.

This "weird" behavior that also happens when the eager mode is enabled is due to the different way the `__eq__` operator for the `tf.Tensor` objects have been overridden.

There is a [question on StackOverflow](https://stackoverflow.com/questions/46785041/why-does-tensorflow-not-override-eq) and a related [Github issue](https://github.com/tensorflow/tensorflow/issues/9359) about this. In short: the `__eq__` operator has been overridden, but the operator **does not** use `tf.equal` to check for the Tensor equality, it just checks for the **Python variable identity** (if you are familiar with the Java programming language, this is precisely like the `==` operator used on `string` objects).
The reason is that the `tf.Tensor` object needs to be hashable since it is used everywhere in the TensorFlow codebase as key for `dict` objects.

OK then, to solve it is required to do not rely upon the `__eq__` operator but use `tf.equal` to check if the equality holds.

However, something should still sound strange: why when invoking the graph-converted function, passing the same `tf.Tensor` `x`, the execution produces `a < b 1 1` instead of `a == b 1 1` as it happens in eager execution?

### Lesson 2: how AutoGraph (don't) converts the operators

So far we supposed that AutoGraph is able to translate not only the `if`, `elif`, and `else` statements to the graph equivalent, but also the Python built-in operators like `__eq__`, `__gt__`, and `__lt__`. In practice, this conversion (still?) does not happen at all.

In the previously converted graph-code, the two condititions are expressed as `a > b` and `a == b` and not as function calls to AutoGraph converted functions (`ag__.converted_call(...)`).

In practice, what happens is that the `cond` is always `False`. We can verify this assertion by adding an additional `elif` to the previous function and calling it again.

```python
@tf.function
def if_elif(a, b):
  if a > b:
    tf.print("a > b", a, b)
  elif a == b:
    tf.print("a == b", a, b)
  elif a < b:
    tf.print("a < b", a, b)
  else:
    tf.print("wat")
x = tf.constant(1)
if_elif(x,x)
```

Output: **wat**.

Hurray?

### Lesson 3: how to write a function

To have the very same behavior in both eager and graph execution we have to know that:

1. The semantic of the operations matters.
2. There are operators that have been overridden following a different semantic (respect to the most natural one, common in Python).
3. AutoGraph converts Python statements naturally (`if`, `elif`, ...) but it requires some extra care when designing a function that is going to be `tf.function` decorated.

In practice, and this is the most important lesson, **use the TensorFlow operators explicitly everywhere** (in the end, the Graph is still present, and we are building it!).

Thus, we can write the correctly eager and graph-convertible function by using the correct `tf.` methods.

```python
@tf.function
def if_elif(a, b):
  if tf.math.greater(a, b):
    tf.print("a > b", a, b)
  elif tf.math.equal(a, b):
    tf.print("a == b", a, b)
  elif tf.math.less(a, b):
    tf.print("a < b", a, b)
  else:
    tf.print("wat")
```

The generated graph code now looks like (removed long parts for clarity):
```python
def tf__if_elif(a, b):
    cond_2 = ag__.converted_call("greater", ...)  # a > b

    def if_true_2():
        ag__.converted_call("print", ...)  # tf.print a > b
        return ag__.match_staging_level(1, cond_2)

    def if_false_2():
        cond_1 = ag__.converted_call("equal", ...)  # tf.math.equal

        def if_true_1():
            ag__.converted_call("print", ...)  # tf.print a == b
            return ag__.match_staging_level(1, cond_1)

        def if_false_1():
            cond = ag__.converted_call("less", ...)  # a < b

            def if_true():
                ag__.converted_call("print", ...)  # tf.print a < b
                return ag__.match_staging_level(1, cond)

            def if_false():
                ag__.converted_call("print", ...)  # tf.print wat
                return ag__.match_staging_level(1, cond)

            ag__.if_stmt(cond, if_true, if_false, get_state, set_state)
            return ag__.match_staging_level(1, cond_1)

        ag__.if_stmt(cond_1, if_true_1, if_false_1, get_state_1, set_state_1)
        return ag__.match_staging_level(1, cond_2)

    ag__.if_stmt(cond_2, if_true_2, if_false_2, get_state_2, set_state_2)
```

Now that every single part of the function has been converted (note the `ag__converted_call` everywhere) the function works as we want, also when it is converted to its graph representation.

## for ... in range

Following the previous 3 lessons, writing a function that uses a `for` loop is trivial. To be entirely sure that the code is correctly graph-converted, we can design the function by using the tensorflow `tf.` methods to help the conversion. So, for a simple function that sums the number from `1` to `X-1` the correct way of designing it is to use:

1. An external `tf.Variable` since the function creates a state and from [part 1](/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/) we know how to deal with it.
2. Use `tf.range` instead of `range` since `tf.range` exists and therefore it is just better to use it.

```python
x = tf.Variable(1)
@tf.function
def test_for(upto):
  for i in range(upto):
    x.assign_add(i)

x.assign(tf.constant(0))
test_for(tf.constant(5))
print("x value: ", x.numpy())
```

The value of the `x` variable is 10, as expected.

The reader is invited to convert the function to its graph representation and check if every statement has been correctly converted.

**Question** (please feel free to answer in the comment section!): what happens if the line `x.assign_add(1)` is replaced by `x = x + i`?

## Conclusions

Writing functions that work correctly in both eager mode and their graph-converted representation requires to know some subtleties that in this three-part particle have been highlighted. To summarize them:

- Functions that create a state need a dedicated design since in eager mode they just work while when converted the stateful objects can create problems. ([part 1](/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/))
- AutoGraph **does not** perform the boxing of the Python native type, and this can slow down the execution **a lot** ([part 2](/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/)); use `tf.Tensor` whenever possible!
- `tf.print` and `print` are different objects; there is a clear distinction between the first call (AutoGraph + function execution + tracing) and any other call of the graph-converted function ([part 2](/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/)).
- The operator overloading of `tf.Tensor` has its own peculiarities. In order to be 100% confident of your function design, and making it also work when it is graph-converted, I highly recommend to use the TensorFlow operators explicitly (call `tf.equal(a,b)` instead of `a == b` and so on).

## Announcement

The article is finished, but I hope to say something pleasing by announcing that I'm authoring my first book about TensorFlow 2.0 and Neural Networks!

> **Hands-On Neural Networks with TensorFlow 2.0**
>
> *Understand TensorFlow, from static graph to eager execution, and design neural networks*

The book is divided into two parts: the first part is more theoretical and is about machine learning and neural networks, with a focus on the intuitive idea behind the presented concepts. The second part, that's the main topic of the book, is about the TensorFlow architecture (from 1.x to 2.0) followed by the implementation of several neural-networks-based solutions to challenging machine learning problems, all using TensorFlow 2.0.

If you want to receive an email when the book is out and also stay up-to-date with the latest articles, leave your email in the form below!
