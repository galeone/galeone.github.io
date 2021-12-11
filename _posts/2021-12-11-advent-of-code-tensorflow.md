---
layout: post
title: "Advent of Code 2021 in pure TensorFlow - day 1"
date: 2021-12-11 08:00:00
categories: tensorflow
summary: "Solving a coding puzzle with TensorFlow doesn't mean throwing fancy machine learning stuff (without any reason) to the problem for solving it. On the contrary, I want to demonstrate the flexibility - and the limitations - of the framework, showing that TensorFlow can be used to solve any kind of problem and that the produced solutions have tons of advantages with respect to the solutions developed using any other programming languages."
authors:
    - pgaleone
---

Even if it's a bit late<sup>\*</sup> I decided to start solving all the puzzles of the [Advent of Code](https://adventofcode.com/) (AoC) using TensorFlow.

One may be wondering why solving the AoC puzzles using TensorFlow and what does it mean to solve them with TensorFlow. First of all, it can be a nice challenge for myself, to see if I'm able to design pure TensorFlow solutions for these challenges.
But perhaps more importantly than the personal challenge, there is the demonstration of the power of the framework. In fact, solving a coding puzzle with TensorFlow doesn't mean throwing fancy machine learning stuff (without any reason) to the problem to solve it. On the contrary, I want to demonstrate the flexibility - and the limitations - of the framework, showing that TensorFlow can be used to solve any kind of problem and that the produced solutions have tons of advantages with respect to the solutions developed using any other programming languages.
In fact, I see TensorFlow more as a programming language than as a mere framework. This is a strong statement, but it's justified (IMHO) by the different way of reasoning one must follow when designing pure-TensorFlow programs.

TensorFlow programs are self-contained descriptions of computation. The inference of a trained machine learning model is a TensorFlow program, but the framework is so flexible that allows describing - and exporting! - generic programs.

I'll try to write a short article for every problem I solve using TensorFlow (so this is the beginning of a series, I hope!), highlighting the peculiarities of the provided solutions and explaining how to reason when creating TensorFlow programs.

<small><sup>\*</sup>Today (12 Dec 2021) is the 11th day of the [Advent of Code](https://adventofcode.com/) in my timezone (UTC+1).</small>

## Advent of Code

To give a bit of context, let's recap what AoC is. From the [about section](https://adventofcode.com/2021/about):

> Advent of Code is an Advent calendar of small programming puzzles for a variety of skill sets and skill levels that can be solved in any programming language you like.

This is pretty much what we need to know. Every year, from the 1st to the 25th of December we do have a puzzle to solve. As I said, I'm a bit late to the party since I decided to start this AoC journey only today, but after all, every cloud has a silver lining. In fact, I [can](https://adventofcode.com/2021/about#:~:text=Can%20I%20stream,streaming%20is%20fine.) safely stream all the solutions I write without ruining the challenge to anyone, since the various daily leaderboards are all full.

## [Day 1: Sonar Sweep](https://adventofcode.com/2021/day/1): part one

You can click on the title above to read the full text of the puzzle. The TLDR version is:

You are given with a set of numbers like
```
199
200
208
210
200
207
240
269
260
263
```

the puzzle goal is count the number of times there's an increase. In the example above, the changes are
```
199 (N/A - no previous measurement)
200 (increased)
208 (increased)
210 (increased)
200 (decreased)
207 (increased)
240 (increased)
269 (increased)
260 (decreased)
263 (increased)
```

so the answer to the puzzle is **7**.

We have all the information required to solve the task. On the AoC page, we can download the real input, that's a text file with the same format presented above, that we can put next to the TensorFlow program we are implementing.

### Design phase

The nature of this task is sequential. We should read the file line by line, convert the line into a number, and compare the i-th number with the (i-1)th. If there's an increase, add one to the counter.

Hence we need a counter, and another variable to keep track of the (i-1)th number, while we process the i-th. Having variables means that our TensorFlow program needs to create a state. Every time the word state occurs during the design phase of a TensorFlow program, [this article](https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/#handling-states-breaking-the-function-scope) should come to mind.

A state is nothing but a `tf.Variable`, but a `tf.Variable` in a TensorFlow program (that, I want to stress to the reader, is a description of the computation) is a node in a graph, that should be declared once. A `tf.Variable` behaves like a Python variable when in **eager mode**, but the TensorFlow programs are (and must) be executed in **graph mode**, and the `tf.Variable` are nodes that can be declared once and repeatedly used.


### Input pipeline

Being sequential, we can use all the input pipelines offered by TensorFlow to efficiently read the data, process, and loop over it.


```python
dataset = tf.data.TextLineDataset("input").map(
    lambda string: tf.strings.to_number(string, out_type=tf.int64)
)
```

Straightforward. We read the "input" file using the `TextLineDataset` object. This `tf.data.Dataset` specialization automatically creates a new element for every new line in the file. Through the `map` method, we apply the conversion from `tf.string` to `tf.int64`.

TensorFlow, differently from Python, is strictly statically typed. Every operation should be performed over the same types, even implicit conversions are not allowed, hence casts and type conversions must be widely used.

### Counting increments

```python
class IncreasesCounter(tf.Module):
    """Stateful counter. Counts the number of "increases"."""

    def __init__(self):
        self._count = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._prev = tf.Variable(0, trainable=False, dtype=tf.int64)

    @tf.function
    def __call__(self, dataset: tf.data.Dataset) -> tf.Tensor:
        """
        Args:
            dataset: the dataset containing the ordered sequence of numbers
                     to process.
        Returns:
            The number of increases. tf.Tensor, dtype=tf.int64
        """
        self._prev.assign(next(iter(dataset.take(1))))
        for number in dataset.skip(1):
            if tf.greater(number, self._prev):
                self._count.assign_add(1)
            self._prev.assign(number)
        return self._count
```

The `IncreaseCounter` class is a complete TensorFlow program. In the `init` we declare and initialize the status variables, and in the `__call__` method we implement the logic required by the puzzle. Note that to be a TensorFlow program, the method must be decorated with `@tf.function`.

Peculiarities:

1. The assignment **must** be performed using the `assign` method. Using the `=` operator will overwrite the Python variable, and not perform the assignment operation in the graph!
1. All the comparisons like `>` are better written using their TensorFlow equivalent (e.g `tf.greater`). Autograph can convert them (you could write `>`), but it's less idiomatic and I recommend to do not relying upon the automatic conversion, for having full control.
1. Extracting the first element from the dataset it's a bit "strange" since it requires to
    - Create a dataset with a single element `.take(1)`
    - Create an iterator from the dataset object (`iter`)
    - Call `next` over the iterator to extract the element.
2. To skip the element assigned in the `self._prev` variable before the loop execution, we need to create another dataset that starts from the second element, by calling `skip(1)` on the dataset object.

### Execution

```python
counter = IncreasesCounter()
increases = counter(dataset)
tf.print("[part one] increases: ", increases)
```

Just create an instance of the `IncreaseCounter` and call it over the dataset previously created. Note that we do use `tf.print` and not `print`. `tf.print` is the operation to use, because it works also in graph mode, while `print` is executed only during the tracing phase (and in eager mode, which we don't want to use).


The execution gives the correct result :) and this brings us to part 2.

## [Day 1: Sonar Sweep](https://adventofcode.com/2021/day/1): part two

TLDR: instead of considering the single values, consider a tree-numbers sliding window. The example above now becomes:
```
199  A
200  A B
208  A B C
210    B C D
200  E   C D
207  E F   D
240  E F G
269    F G H
260      G H
263        H
```

Note: the input doesn't change, these A B C, and so on are here only to visualize the sliding windows. The goal is to sum all the numbers in a window (e.g. 199+200+208 for windows `A`) and compare the sum with the sum of the previous sliding window.

```
A: 607 (N/A - no previous sum)
B: 618 (increased)
C: 618 (no change)
D: 617 (decreased)
E: 647 (increased)
F: 716 (increased)
G: 769 (increased)
H: 792 (increased)
```

In this case the answer is **5**.

### Design phase - part two

We already have a TensorFlow program that can detect increases of adjacent numbers in a dataset. So, we can just feed to the program a different output to get the correct result.

TensorFlow offers us several functions for working with datasets. As we've seen, it is possible to skip elements and create a new dataset, apply transformation function with `map`, and so on. Thus, we can solve this challenge by creating a dataset of the resulting sums of the various sliding windows.

### Input pipeline - part two

The idea is to create 3 different datasets by shifting by 1 element every time. Create batches of 3 elements (the sliding windows), and them sum all the values in the batch (window).

The 3 datasets can be merged interleaving the values, using the order `1,2,3`, `1,2,3` and so on. This order means: pick the i-th element from the first dataset, then pick the i-th element from the second dataset, then pick the i-th element from the third dataset, then increment i. Repeat until all the dataset consumed all the elements.

```python
datasets = [dataset, dataset.skip(1), dataset.skip(2)]
for idx, dataset in enumerate(datasets):
    datasets[idx] = dataset.batch(3, drop_remainder=True).map(tf.reduce_sum)

interleaved_dataset = tf.data.Dataset.choose_from_datasets(
    datasets, tf.data.Dataset.range(3).repeat()
)
```

### Execution - part two

The instance `counter` of `IncreaseCounter` has a state, hance we can't re-use it because we haven't added a method to reset the state. Thus, we need to create a new instance and pass the `interleaved_dataset` to get the correct result.

```python
counter = IncreasesCounter()
increases = counter(interleaved_dataset)
tf.print("[part two] increases: ", increases)
```

The first puzzle is completely solved :)

## Conclusion

Solving the AoC puzzles with TensorFlow can be not just fun (come on, it's a nice challenge thinking about all these nuances :)), but it can be also a good way to design very efficient solutions. In fact, I still haven't spent some word about the advantages of these implementations, but there are many.

- The solution can run on any hardware. If you have a supported GPU, it runs on it.
- Any operation that can be executed in parallel, is automatically parallelized by the framework.
- These solutions are **language agnostic**. Yes, we designed and executed them in Python, but we could export them as [SavedModel](https://www.tensorflow.org/api_docs/python/tf/saved_model), and re-use the same logic in any other programming language since all we need is the [TensorFlow C runtime](https://www.tensorflow.org/install/lang_c). For example, a SavedModel of this program can be executed in Go using [tfgo](https://github.com/galeone/tfgo).

I'm doing this for fun (and I'm having fun, really), so expect another article for day 2 coming soon!

For any feedback or comment, please use the Disqus form below - thanks!

PS: I'm posting all the solutions also on GitHub, you can find them here: [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).
