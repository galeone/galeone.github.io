---
layout: post
title: "Advent of Code 2022 in pure TensorFlow - Day 6"
date: 2022-12-27 08:00:00
categories: tensorflow
summary: "Solving problem 6 of the AoC 2022 in pure TensorFlow allows us to understand how powerful this framework can be. In particular, problem 6 can be solved with a highly efficient and parallel solution, using just a single feature of tf.data.Dataset: interleave."
authors:
    - pgaleone
---

Solving problem 6 of the AoC 2022 in pure TensorFlow allows us to understand how powerful this framework can be. In particular, problem 6 can be solved with a highly efficient and parallel solution, using just a single feature of `tf.data.Dataset`: interleave.

## [Day 6: Tuning Trouble](https://adventofcode.com/2022/day/6)

You can click on the title above to read the full text of the puzzle. The TLDR version is: we need to decode a signal. The signal is a string containing some "random" characters. Decoding a signal means detecting a marker character. A marker character is defined as the first character of a sequence of 4 (part 1) or 14 (part 2) characters without repeater characters inside.

So, given a puzzle input like
```
mjqjpqmgbljsphdztnvjfqwrcgsmlb
```

we need to analyze the signal sequentially (left to right) and search for the first sequence of 4 characters that are all different. In this case, we start from the left `mjq` are the first 3 characters. The 4 characters, however, is a `j` that's contained in the `mjq` sequence, so `j` is repeated and thus `m` is not a marker character. The first time a marker appears is after the seventh character arrives. In this case, the last four characters received are `jpqm`, which are all different. Thus, the result of the analysis is `7`.

Part 1 asks us to detect the marker character considering sequences of `4` different characters, part 2 instead requires sequences of `14` different characters.

### Design Phase

The problem may look complicated since it requires searching for sequences of different characters on strings that can potentially overlap. For example, given the sample input

```
mjqjpqmgbljsphdztnvjfqwrcgsmlb
```

The first search fails. `mjqj` is not a valid sequence. Thus, we need to restart the search from the first `j` character of the sequence, finding `jqjp` that's once again not correct. We need to repeat this very same algorithm until we don't find the `jpqm` string that satisfies the condition.

There's a thing to note that will help in designing a fast solution for this problem: every search is potentially independent of each other. If we can split the input sequence into various sub-strings like (for part 1, 4 splits, for part 2, 16 splits):

- `[0,4] -> [4,8] -> [8,12] -> ...`
- `[1,5] -> [5,9] -> [9-13] -> ...`
- `[2,6] -> [6,10] -> [10-14] -> ...`
- `[3,7] -> [7,11] -> [11-15] -> ...`

and *interleave* the sub-strings generating the sequence `[0,4] -> [1,5] -> [2,6] -> [3,7] -> [4,8] -> ...`, we can loop over this sequence and stop when the correct substring meets the criteria (all the characters are different).

### Understanding tf.data.Dataset interleave

[tf.data.Dataset.interleave](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave) is the superhero of data transformation. This is the method signature

```python
interleave(
    map_func,
    cycle_length=None,
    block_length=None,
    num_parallel_calls=None,
    deterministic=None,
    name=None
)
```

The interleave method allows us to apply a transformation (`map_func`) to an input dataset, **generate a new dataset for every iteration**, control the behavior of every dataset, and interleave the results into a single output stream of a new dataset object.

The `cycle_length` and `block_length` arguments control the order in which elements are produced. The `num_parallel_calls` and `deterministic` parameters control the multi-thread behavior of the transformation. When `num_parallel_calls` is specified, the `cycle_lenght` elements produced from the initial dataset, are processed by `num_parallel_calls` threads. This processed data is then grouped in `block_length` elements and produced as output.

In short, you can think about the `block_length` parameter as the number of elements that the interleaved dataset will produce on every iteration, while `cycle_length` is the number of elements for every generated dataset that will be processed concurrently. You can specify the concurrency level through the `num_parallel_calls` parameter and with the `deterministic` parameter you can control that every iteration of the dataset respects your deterministic, intended, behavior. In our case, we are interested in having a deterministic approach, since the position of the marker character is important, but of course, there are problems in which you just want to apply transformations to datasets and interleave the results, without being interested in the order of the interleaving.

### Solving the problem

`tf.data.Dataset.interleave` is all we need to solve this problem. With a correct configuration, it can model *exactly* the behavior described in the [design phase](#design-phase) section.

The dataset, however, requires to be converted from a single long string (the input signal) to a real "stream" of characters, that we can use as input dataset for our interleave transformation.

```python
chars = tf.convert_to_tensor(
    next(
        dataset.map(lambda line: tf.strings.bytes_split(line))
        .take(1)
        .as_numpy_iterator()
    )
)

dataset = tf.data.Dataset.from_tensors(tf.reshape(chars, [-1, 1])).unbatch()
```

`dataset` now is a `tf.data.Dataset` that produces characters on every iteration (a real stream!). So, how can we create an interleaved version of this dataset that produces the sequence of sub-strings we are interested in?

We should be able to produce 4 (or 16 for part 2) new datasets, each of them starting from a different offset.

- Dataset 1. Offset 0: `mjqj` - `pqmg` - `bljs` ...
- Dataset 2: Offset 1: `jqjp` - `qmgb` - `ljsp` ...
- Dataset 3: Offset 2: `qjpq` - `mgbl` - `jsph` ...
- Dataset 4: Offset 3: `jpqm` - `gblj` - `sphd` ...

Using the `interleave` method is quite easy: we just need to create the right dataset of offsets and generate the interleaved datasets. This dataset will be then used by the interleave method, as specified by its configuration, to produce the desired result.

```python
interleaved = tf.data.Dataset.range(4).interleave(
    lambda offset: dataset.skip(offset).batch(4),
    cycle_length=4,
    block_length=1,
    num_parallel_calls=4,
    deterministic=True,
)
```

Yes, it really is that easy! With `tf.data.Dataset.range(4)` we are generating the dataset that produces the values from 0 to 4 sequentially. This dataset is used to produce the `offset` value for the `dataset.skip` method invoked as the transformation to the input dataset. So, our `map_func` produces a new `tf.data.Dataset` on every iteration of the range-dataset. Every dataset then extracts a batch of 4 elements (the substrings).

The configuration, allows us to iterate over the interleaved 4 datasets, in a deterministic way, extracting on every iteration a batch of 4 elements for each created dataset, interleaved as we expect.

Thus, to completely solve the problem we have to loop over this dataset, check for the uniqueness of the elements in the loop, and get the char's index:

```python
for count, b in enumerate(interleaved):
    y, _ = tf.unique(tf.reshape(b, -1))
    if tf.equal(tf.shape(y)[0], 4):
        tf.print(y)
        # 1: starts from 0
        # 3: the remaining chars in the sequence
        tf.print("unique found at char: ", count + 4)
        break
```

Here we go, day 6 problem solved in pure TensorFlow! Solving part 2 is identical, just replace every occurrence of 4 with 14.

Give a look at the complete [solution](https://github.com/galeone/tf-aoc/blob/main/2022/6/main.py).

## Conclusion

You can see the complete solutions in folder `6` in the dedicated GitHub repository (in the `2022` folder): [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

Solving problem 6 allowed us to use a very powerful feature of `tf.data.Dataset`: interleave. In a few lines, this method allows us to define a complete, highly parallel, and efficient data transformation pipeline, that allows us to transform and group data gathered from different datasets. The expressive power of this method, moreover, allowed us to solve the problem in a very elegant way IMHO.

If you missed the article about the previous days’ solutions, here's a handy list

- [Advent of Code 2022 in pure TensorFlow - Days 1 & 2](/tensorflow/2022/12/04/advent-of-code-tensorflow-day-1-and-2/).
- [Advent of Code 2022 in pure TensorFlow - Days 3 & 4](/tensorflow/2022/12/11/advent-of-code-tensorflow-day-3-and-4/).
- [Advent of Code 2022 in pure TensorFlow - Day 5](/tensorflow/2022/12/21/advent-of-code-tensorflow-day-5/)

For any feedback or comment, please use the Disqus form below - thanks!
