---
layout: post
title: "Advent of Code 2022 in pure TensorFlow - Day 5"
date: 2022-12-18 08:00:00
categories: tensorflow
summary: "Differently from the previous 2 articles, where I merged the description of the solutions of two problems into one article, this time the whole article is dedicated to the pure TensorFlow solution of the problem number 5. The reason is simple: solving this problem in pure TensorFlow hasn't been straightforward and for this reason it is worth explaining all the limitations and the subtle bugs (?) found during the solution."
authors:
    - pgaleone
---

Differently from the previous 2 articles, where I merged the description of the solutions of two problems into one article, this time the whole article is dedicated to the pure TensorFlow solution of the problem number 5. The reason is simple: solving this problem in pure TensorFlow hasn't been straightforward and for this reason it is worth explaining all the limitations and the subtle bugs (?) found during the solution.

In the first part of the article I'll explain the solution that solves completely both parts of the puzzle. In the second part, instead, I'll analyze the problem I encountered in the first design of the solution where I wanted to solve the puzzle using a `tf.Variable` with an "undefined shape". Feature available for every `tf.Variable` but not clearly documented (IMHO). So, at the end of this article, we'll understand a little bit more about what happend when the `validate_shape` argument of `tf.Variable` is set to `False`.


## [Day 5: Supply Stacks](https://adventofcode.com/2022/day/5)

You can click on the title above to read the full text of the puzzle. The TLDR version is: we have an initial configuration of crates (the puzzle input) and a set of moves to perform. Part 1 constrains the crane that's moving the crates to pick a single crate at a time, while part 2 allows multiple crates to be picked up at the same time. The problem asks us to determine, after having moved the crates, what crate ends up on top of each stack.

So, given a puzzle input like
```
    [D]
[N] [C]
[Z] [M] [P]
 1   2   3

move 1 from 2 to 1
move 3 from 1 to 3
move 2 from 2 to 1
move 1 from 1 to 2
```

Part 1, that wants us to move the crates one at the time, ends up with this final configuration

<pre>
        [<b>Z</b>]
        [N]
        [D]
[<b>C</b>] [<b>M</b>] [P]
 1   2   3
 </pre>

Thus, the result is: "CMZ".

For part 2 instead, where multiple crates can be picked up at the same time, the final configuration is

<pre>
        [<b>D</b>]
        [N]
        [Z]
[<b>M</b>] [<b>C</b>] [P]
 1   2   3
 </pre>

in this case the result is "MCD".

### Design Phase

The problem can be breakdown into 4 simple steps:

1. Read the first part of the input: parse the crates
1. Create a data structure that models the stacks of crates
1. Read the second part of the input: parse the instructions
1. Iteratively transform the previously created data structure according to the instructions

As anticipated in the [previous article](/tensorflow/2022/12/04/advent-of-code-tensorflow-day-1-and-2/#input-pipeline) the input pipeline will never change in the AoC problems, thus this part won't be presented in the article. It is, and always will be, a dataset that produces `tf.string` items (every single line read from the input).

### Splitting strings: ragged tensors

The [`tf.strings`](https://www.tensorflow.org/api_docs/python/tf/strings/) module contains several utilities for working with `tf.Tensor` with `dtype=tf.string`. For splitting in two identical halves every line, the `tf.strings.substr` function is perfectly suited. Since we want to apply the same transformation to every line of the dataset we can define a function (that, thus, we'll always be executed in graph mode) that process the line and returns the two strings.

```python
@tf.function
def split(line):
    length = tf.strings.length(line) // 2
    position = length

    return tf.strings.substr(line, pos=0, len=length), tf.strings.substr(
        line, pos=position, len=length
    )
```

The code is trivial. The only thing to note is that strings are a variable-length data type. Thus, functions like `substr`, or `split` cannot return a `tf.Tensor` since a `tf.Tensor` is always made of **identical** dimensions (e.g. every element in a Tensor has a well-defined shape). Instead, these functions work with [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor?hl=en)s as input or output.

Applying this transformation is just the invocation of the `.map` method over the input dataset. 

```python
splitted_dataset = dataset.map(split)
```

The `splitted_dataset` is a `tf.data.Dataset` whose elements are 2 ragged tensors.

### Items to priority: StaticHashTable

The mapping between items and the corresponding priorities is known in advance (and not known at runtime), thus we can create a lookup table that given a character, it returns the corresponding priority. The perfect tool is the [tf.lookup.StaticHashTable`](https://www.tensorflow.org/api_docs/python/tf/lookup/StaticHashTable?hl=en).

TensorFlow's flexibility allows us to create this mapping in a single call. We only need

1. The characters to map (we can write manually the lowercase/uppercase alphabet or get this constant from the python `string` module)
2. The priority values

```python
keys_tensor = tf.concat(
    [
        tf.strings.bytes_split(string.ascii_lowercase),
        tf.strings.bytes_split(string.ascii_uppercase),
    ],
    0,
)
vals_tensor = tf.concat([tf.range(1, 27), tf.range(27, 53)], 0)

item_priority_lut = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), default_value=-1
)
```

we now have a lookup table ready to use. Thus, we can create a `to_priority(first, second)` function that will map every character in `first` and `second` to their corresponding priority.

TensorFlow allows us to do the mapping of every character to its priority in parallel. Using [`tf.strings.byte_split`](https://www.tensorflow.org/api_docs/python/tf/strings/bytes_split) we can pass from a string ("abc..") to a tensor of characters ('a', 'b', 'c', ...).

```python
@tf.function
def to_priority(first, second):
    first = tf.strings.bytes_split(first)
    second = tf.strings.bytes_split(second)
    return item_priority_lut.lookup(first), item_priority_lut.lookup(second)
```

as anticipated, the lookup is done in parallel (the `first` parameter input to `lookup` is a tensor of characters, and the lookup for every character is done in a single pass).

The `to_priority` function returns a pair of `tf.Tensor` containing the corresponding priorities.

Once again, applying the transformation to the dataset is trivial

```python
splitted_priority_dataset = splitted_dataset.map(to_priority)
```

### Finding common priorities: tf.sets and tf.sparse.SparseTensor

Working with sets in TensorFlow is not as simple as working with sets in other languages. In fact, every function in the [`tf.sets`](https://www.tensorflow.org/api_docs/python/tf/sets/) module accepts a particular input data representation.

You cannot pass, for example, two tensors like (1,2,3) and (1,0,0) to the `tf.sets.intersection` function, and expect to get the value `1`. You need to reshape them, making the input parameters behave like an array of sets represented as a sparse tensor (since TensorFlow is designed for executing the same operation in parallel). That's why you'll see the `tf.expand_dims` call in the code below.

```python
@tf.function
def to_common(first, second):
    first = tf.expand_dims(first, 0)
    second = tf.expand_dims(second, 0)
    intersection = tf.sets.intersection(first, second)
    return tf.squeeze(tf.sparse.to_dense(intersection))
```

The `intersection` is a [`tf.sparse.SparseTensor`](https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor) that's a compact representation for a `tf.Tensor`. Usually, this representation is very useful when working with "sparse problems" (e.g. problems dealing with huge matrices/tensors where the majority of the elements are 0). For our problem instead, it's useless and we can get the dense representation from its sparse counterpart (with `tf.sparse.to_dense`), and then return the single common priority as a scalar tensor by squeezing all the dimensions of size 1 from the shape of the dense tensor.

Once again, we transform the dataset

```python
common_elements = splitted_priority_dataset.map(to_common)
```

Perfect! The `common_elements` is a `tf.data.Dataset` that contains the common priority for every line of the original dataset. We can now loop over it (`list`) and covert it to a `tf.Tensor` so we can use the `tf.reduce_sum` for getting the sum of the identified common priorities.

```python
tensor = tf.convert_to_tensor(list(common_elements.as_numpy_iterator()))
tf.print("sum of priorities of common elements: ", tf.reduce_sum(tensor))
```

Part one: ✅

## [Day 3: Rucksack Reorganization](https://adventofcode.com/2022/day/3): part two

The second part of the problem asks us to do not to consider a single rucksack, but a group of 3 rucksacks. Every 3 rucksacks have a single element in common, we need to identify it, find its priority, and get the sum of these newly identified priorities.

The problem can be breakdown into 3 steps:

1. Create the group of characters: using `tf.data.Dataset` is trivial, it means to just call `batch(3)`.
2. Mapping the characters to the priorities
3. Find the common priority for every batch (instead of every line as we did in the previous part) and sum them.

The steps to follow are very similar to part 1, so instead of detailing every single step, we can just go straight to the solution.

```python
# batch
grouped_dataset = dataset.batch(3)

# mapping
grouped_priority_dataset = grouped_dataset.map(
    lambda line: item_priority_lut.lookup(tf.strings.bytes_split(line))
)

@tf.function
def to_common_in_batch(batch):
    intersection = tf.sets.intersection(
        tf.sets.intersection(
            tf.expand_dims(batch[0], 0), tf.expand_dims(batch[1], 0)
        ),
        tf.expand_dims(batch[2], 0),
    )
    return tf.squeeze(tf.sparse.to_dense(intersection))

grouped_common_elements = grouped_priority_dataset.map(to_common_in_batch)
tensor = tf.convert_to_tensor(list(grouped_common_elements.as_numpy_iterator()))
tf.print("sum of priorities of grouped by 3 elements: ", tf.reduce_sum(tensor))
```

Here we go, day 3 problem solved!

The day 4 problem is quite easy and, from the TensorFlow point of view, it doesn't use new functionalities. So it's not worth writing a dedicated article about it but I'll try to summarize the main peculiarities in the section below.

## [Day 4: Camp Cleanup](https://adventofcode.com/2022/day/4): part one

You can click on the title above to read the full text of the puzzle. The TLDR version is: given a list of pairs of ranges (the puzzle input) we need to count in how many pairs one range **fully contains** the other.

The puzzle input is something like

```
2-4,6-8
2-3,4-5
5-7,7-9
2-8,3-7
6-6,4-6
2-6,4-8
```

Thus, the first pair is made of the range (2,3,4) and (6,7,8). They have no elements in common and thus it doesn't satisfy the requirement. The ranges 2-8 and 3-7, instead satisfy this requirement since 3-7 (3,4,5,6,7) is fully contained in 2-8 (2,**3,4,5,7**,8).

Problem breakdown:

1. Parse the data and get the start and end number for every range in every pair
2. Just filter (`tf.data.Dataset.filter`) the dataset, and keep only the elements that satisfy the condition.

```python
pairs = dataset.map(lambda line: tf.strings.split(line, ","))
ranges = pairs.map(
    lambda pair: tf.strings.to_number(tf.strings.split(pair, "-"), tf.int64)
)

contained = ranges.filter(
    lambda pair: tf.logical_or(
        tf.logical_and(
            tf.math.less_equal(pair[0][0], pair[1][0]),
            tf.math.greater_equal(pair[0][1], pair[1][1]),
        ),
        tf.logical_and(
            tf.math.less_equal(pair[1][0], pair[0][0]),
            tf.math.greater_equal(pair[1][1], pair[0][1]),
        ),
    )
)
```

It's really just a filter function over the element of a dataset. To solve the problem, we just need to count the elements of this dataset. We can do it by looping over it, converting the result to a `tf.Tensor` and get its outer dimension.

```python
pairs = dataset.map(lambda line: tf.strings.split(line, ","))
ranges = pairs.map(
    lambda pair: tf.strings.to_number(tf.strings.split(pair, "-"), tf.int64)
)

contained = ranges.filter(
    lambda pair: tf.logical_or(
        tf.logical_and(
            tf.math.less_equal(pair[0][0], pair[1][0]),
            tf.math.greater_equal(pair[0][1], pair[1][1]),
        ),
        tf.logical_and(
            tf.math.less_equal(pair[1][0], pair[0][0]),
            tf.math.greater_equal(pair[1][1], pair[0][1]),
        ),
    )
)
contained_tensor = tf.convert_to_tensor(
    list(iter(contained.map(lambda ragged: tf.sparse.to_dense(ragged.to_sparse()))))
)
tf.print("Fully contained ranges: ", tf.shape(contained_tensor)[0])
```

Part 1 completed!

## [Day 4: Camp Cleanup](https://adventofcode.com/2022/day/4): part two

The second part of the problem simply asks to find the number of ranges that **partially** overlap (e.g. 5-7,7-9 overlaps in a single section (7), 2-8,3-7 overlaps all of the sections 3 through 7, and so on...).

The solution is even easier than the previous part.

```python
overlapping = ranges.filter(
    lambda pair: tf.logical_not(
        tf.logical_or(
            tf.math.less(pair[0][1], pair[1][0]),
            tf.math.less(pair[1][1], pair[0][0]),
        )
    )
)

overlapping_tensor = tf.convert_to_tensor(
    list(
        iter(overlapping.map(lambda ragged: tf.sparse.to_dense(ragged.to_sparse())))
    )
)

tf.print("Overlapping ranges: ", tf.shape(overlapping_tensor)[0])
```

That's all, day 4 problem is completely solved in pure TensorFlow!

## Conclusion

You can see the complete solutions in folders `3` and `4` in the dedicated GitHub repository (in the `2022` folder): [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

Solving these 2 problems has been straightforward and, in practice, both solutions are just transformations of every line of the input dataset to something else, until we end up with a single `tf.Tensor` containing the result we are looking for.

So far, TensorFlow has demonstrated to be flexible enough for solving these simple programming challenges. Anyway, I've already solved other problems, and some of them will show the limitations of using TensorFlow as a generic programming language (and perhaps, I found some bugs!).

If you missed the article about the previous days’ solutions, here's a link: [Advent of Code 2022 in pure TensorFlow - Days 1 & 2](/tensorflow/2022/12/04/advent-of-code-tensorflow-day-1-and-2/).

For any feedback or comment, please use the Disqus form below - thanks!
