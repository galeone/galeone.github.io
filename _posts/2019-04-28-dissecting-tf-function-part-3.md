---
layout: post
title: "Analyzing tf.function to discover AutoGraph strengths and subtleties - part 3"
date: 2019-04-28 08:00:00
categories: tensorflow tf.function
summary: "In part 1 we learned how to convert a 1.x code to its eager version, the eager version to its graph representation and faced the problems that arise when working with functions that create a state. In this second part, we’ll analyze what happens when instead of a tf.Variable we pass a tf.Tensor or a Python native type as input to a tf.function decorated function. Are we sure everything is going to be converted to the Graph representation we expect?"
---

In [part 1](/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/) we learned how to convert a 1.x code to its eager version, the eager version to its graph representation and faced the problems that arise when working with functions that create a state.

In [part 2](/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/) we learned that `tf.function` creates a new graph for every different input value, if the input type is not a `tf.Tensor` object and that this could slow down (or speed up if correctly used) the execution. Moreover, the differences between the `tf.autograph` generated source code and what happens, instead, when using AutoGraph trough `tf.function` have been highlighted.

In this third and last part, we'll analyze what happens when `tf.function` is used to convert a function that contains complex Python constructs in its body. Should we design function thinking about how they are going to be converted?(spoiler: yes).

## AutoGraph capabilities and limitations

In the Tensorflow repository, in the `python/autograph` folder, we can find [a document](https://github.com/tensorflow/tensorflow/blob/560e2575ecad30bedff5b192f33f6d06b19ccaeb/tensorflow/python/autograph/LIMITATIONS.md) that explains which are the capabilities and the limitations of the AutoGraph module together with a list of the Python constructs it is able to convert.

The [table](https://github.com/tensorflow/tensorflow/blob/560e2575ecad30bedff5b192f33f6d06b19ccaeb/tensorflow/python/autograph/LIMITATIONS.md#python-language-support-status) in the section Python Language Support Status contains a list of all the Python constructs that AutoGraph explicitly supports, plan to support, or won't support. Among them, we can find the widely used `while`, `for`, `if`  statements, the Python built-in `print`, `len`, `range`, and the iterator construct.

In the next sections various Python functions that uses these Python constructs are analyzed, in order to find if the function body gets converted as we expect or it is required to design the functions thinking about the graph conversion.

## if ... else

Here's the function we are going to analyze:

```python

```

## if ... elsif ... else

## for ... in range

## while


## Conclusions

