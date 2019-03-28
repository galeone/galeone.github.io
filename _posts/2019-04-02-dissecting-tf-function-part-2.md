---
layout: post
title: "Analyzing tf.function to discover AutoGraph strengths and subtleties - part 2"
date: 2019-03-21 08:00:00
categories: tensorflow tf.function
summary: ""
---

In [part 1](/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/) we learned how to convert a 1.x code to its eager version, the eager version to its graph representation and concluded analyzing the problems is need to face when working with functions that create a state.

In this second part, we’ll study what happens when instead of a `tf.Variable` we pass a `tf.Tensor` or a Python type as input to a `tf.function` decorated function, together with the analysis of the AutoGraph behavior when the Python code is executed in the first function call: are we sure everything is going to be converted to the Graph representation we expect?

## Changing tf.Tensor input type

Let's start by defining our Python function. The function parameters type is of fundamental importance since is used to create a graph, that is a statically typed object, and to assign an ID to it (for a complete and informal explanation of what's going on when calling the function for the first time, see [tf.function: layman explanation](/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/#tffunction-layman-explanation)).

```python
@tf.function
def f(x):
    print("Python execution: ", x)
    tf.print("Graph execution: ", x)
    return x
```

The Python function accepts a variable `x` that can be every value that can be printed.


## Conclusions

If you find this article useful, feel free to share it using the buttons below!
