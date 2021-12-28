---
layout: post
title: "Advent of Code 2021 in pure TensorFlow - day 7"
date: 2021-12-28 08:00:00
categories: tensorflow
summary: "The day 7 challenge is easily solvable with the help of the TensorFlow ragged tensors. In this article, we'll solve the puzzle while learning what ragged tensors are and how to use them."
authors:
    - pgaleone
---

The day 7 challenge is easily solvable with the help of the [TensorFlow ragged tensors](https://www.tensorflow.org/guide/ragged_tensor). In this article, we'll solve the puzzle while learning what ragged tensors are and how to use them.

Part 1 doesn't require ragged tensors, but to easily solve part 2 we'll introduce them briefly.

## [Day 7: The Treachery of Whales](https://adventofcode.com/2021/day/7)

You can click on the title above to read the full text of the puzzle. The TLDR version is:

Our input is a comma-separated list of horizontal positions

```
16,1,2,0,4,2,7,1,2,14
```

Each number represents the position on the X-axis of a crab. Every crab can move along the horizontal plane, and every movement costs 1 unit of fuel.

The text shows us the position that costs the least fuel: `2`.

> - Move from 16 to 2: 14 fuel
> - Move from 1 to 2: 1 fuel
> - Move from 2 to 2: 0 fuel
> - Move from 0 to 2: 2 fuel
> - Move from 4 to 2: 2 fuel
> - Move from 2 to 2: 0 fuel
> - Move from 7 to 2: 5 fuel
> - Move from 1 to 2: 1 fuel
> - Move from 2 to 2: 0 fuel
> - Move from 14 to 2: 12 fuel
>
> This costs a total of **37** fuel.
> This is the cheapest possible outcome; more expensive outcomes include aligning at position 1 (41 fuel), position 3 (39 fuel), or position 10 (71 fuel).

The puzzle goal is to align all the crabs to a position that **minimizes the overall fuel consumption**.

### Design phase: part one

There are two observations to do:

- The smallest possible fuel consumption is 0. Hence, it's likely for the optimal configuration to be **in the neighborhood** of the more crowded position. In fact, in the example, 2 is the most frequent position.
- Minimizing the overall fuel consumption means finding the $x$ value that satisfies the relation $$ \sum_{i} {\left\|p_i - x\right\|} < \sum_{i} {\left\|p_i - y\right\|} \quad \forall x \neq y $$ where $$x$$ and $$y$$ are elements of the position dataset and the sums are performed over all the available positions.

The observation 2 domain can be constrained to something smaller than all the available positions because of observation 1. The correct estimation of the neighborhood size is not required, hence we can just fix it to an arbitrary big value like half the dataset size.


### Input pipeline

We create a `tf.data.Dataset` object for reading the text file line-by-line [as usual](/tensorflow/2021/12/11/advent-of-code-tensorflow/#input-pipeline). Since the dataset is a single line, we can keep it in memory and convert the dataset as a `tf.Tensor` that's easy to use.

```python
dataset = (
    tf.data.TextLineDataset("input")
    .map(lambda string: tf.strings.split(string, ","))
    .map(lambda string: tf.strings.to_number(string, out_type=tf.int64))
    .unbatch()
)

dataset_tensor = tf.convert_to_tensor(list(dataset))
```

### Finding the neighborhood of the most frequent position

[Observation 1](#design-phase-part-one) requires us to

1. Find the most frequent position
1. Decide a neighborhood size
1. Find the neighborhood extremes (the min/max positions)

```python
y, idx, count = tf.unique_with_counts(dataset_tensor, tf.int64)

max_elements = tf.reduce_max(count)
most_frequent_position = y[idx[tf.argmax(count)]]

neighborhood_size = tf.constant(
    tf.shape(dataset_tensor, tf.int64)[0] // tf.constant(2, tf.int64), tf.int64
)

min_neigh_val = tf.clip_by_value(
    most_frequent_position - neighborhood_size,
    tf.constant(0, tf.int64),
    most_frequent_position,
)

max_val = tf.reduce_max(dataset_tensor) + 1
max_neigh_val = tf.clip_by_value(
    most_frequent_position + neighborhood_size,
    most_frequent_position,
    max_val,
)
```

[`tf.unique_with_counts`](https://www.tensorflow.org/api_docs/python/tf/unique_with_counts) in combination with [`tf.argmax`](https://www.tensorflow.org/api_docs/python/tf/math/argmax?hl=en) allows us to find the most frequent position.

The neighborhood size is arbitrary set to half the dataset size, and the neighborhood extremes are constrained into the `[0, max(dataset)]` range.

### Minimizing the cost

[Observation 2](#design-phase-part-one) is precisely the formula to implement. We look around in the neighborhood of the most frequent value, find the cost, and check if it's the minimum cost found.

```python
min_cost = tf.Variable(tf.cast(-1, tf.uint64))
found_position = tf.Variable(-1, dtype=tf.int64)

for x in tf.range(min_neigh_val, max_neigh_val):
    cost = tf.cast(tf.reduce_sum(tf.abs(dataset_tensor - x)), tf.uint64)
    if tf.less(cost, min_cost):
        min_cost.assign(cost)
        found_position.assign(x)
tf.print("(part one) min_cost: ", min_cost, " in position: ", found_position)
```

Pretty standard. The only peculiarity is how the `min_cost` variable has been initialized. Since there are no constants exposed by TensorFlow for the min/max values for integer values we can just define `min_cost` as an `uint64` variable, initialized with a `-1`. -1 is represented as an integer value with all the bits set to 1, and thus if interpreted as an `uint64` it gives us the maximum representable value for an unsigned int at 64 bits.

Part 1 is solved! We are ready for the part 2 challenge.

## Design phase: part 2

The puzzle introduces a slight variation of fuel consumption. It's not more constant (1 fuel unit per position), but it grows linearly with the position change: the first step costs 1, the second step costs 2, the third step costs 3, and so on.

The previous example best position is not `5`, in fact

> - Move from 16 to 5: 66 fuel
> - Move from 1 to 5: 10 fuel
> - Move from 2 to 5: 6 fuel
> - Move from 0 to 5: 15 fuel
> - Move from 4 to 5: 1 fuel
> - Move from 2 to 5: 6 fuel
> - Move from 7 to 5: 3 fuel
> - Move from 1 to 5: 10 fuel
> - Move from 2 to 5: 6 fuel
> - Move from 14 to 5: 45 fuel
>
> This costs a total of 168 fuel. This is the new cheapest possible outcome; the old alignment position (2) now costs 206 fuel instead.

This requirement requires us only to change how our fuel consumption is calculated, the observations performed in the initial design phase are still valid.

In practice, now we can compute the distance from a candidate position as we previously did ($$ d_x = \left\| p_i - x \right\| $$) and use this distance to compute the fuel consumption.

The new fuel consumption is `d_x + d_{x-} + \cdots + 1 + 0`. There are several ways for doing this calculation, one of them is to create tensors with a different number of elements  - depending on the value of the distance - and sum them together. The perfect tool for doing it is using ragged tensors.

### TensorFlow ragged tensors

The [documentation](https://www.tensorflow.org/guide/ragged_tensor) is pretty clear:

> Your data comes in many shapes; your tensors should too. Ragged tensors are the TensorFlow equivalent of nested variable-length lists. They make it easy to store and process data with non-uniform shapes

There's a whole package dedicated to ragged tensors: [`tf.ragged`](https://www.tensorflow.org/api_docs/python/tf/ragged). This package defines ops for manipulating ragged tensors (tf.RaggedTensor), which are tensors with non-uniform shapes. In particular, each RaggedTensor has one or more ragged dimensions, which are dimensions whose slices may have different lengths. For example, the inner (column) dimension of `rt=[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is ragged, since the column slices (`rt[0, :], ..., rt[4, :]`) have different lengths. 

## Ragged range

In the `tf.ragged` package we have the `tf.ragged.range` function that's the perfect candidate for implementing our solution. In fact, given a tensor containing the distances of the point `x` from all the other points of the dataset `[3, 1, 5, ...]` we can create in a single - parallel - step a ragged tensor containing all the fuel consumptions

```
[
    [0 1 2],
    [0],
    [0 1 2 3 4],
    ...
]
```

and sum `1` to all of them for obtaining the lists with the values to sum for obtaining the fuel consumption, and thus the cost.

```python
# -- Part 2 --
min_cost, found_position = tf.cast(-1, tf.uint64), -1
min_cost.assign(tf.cast(-1, tf.uint64))
found_position.assign(-1)
for x in tf.range(min_neigh_val, max_neigh_val):
    diff = tf.abs(dataset_tensor - x)
    lists = tf.ragged.range(tf.ones(tf.shape(diff)[0], dtype=tf.int64), diff + 1)
    cost = tf.cast(tf.reduce_sum(lists), tf.uint64)
    if tf.less(cost, min_cost):
        min_cost.assign(cost)
        found_position.assign(x)

tf.print("(part two) min_cost: ", min_cost, " in position: ", found_position)
```

Using the ragged tensors has been possible to solve part 2 with a single change to part 1.

## Conclusion

You can see the complete solution in folder `7` on the dedicated Github repository: [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

The challenge in the challenge of using only TensorFlow for solving the problem is slowly progressing, so far I solved all the puzzles up to Day 11 (inclusive). So get ready for at least 4 more articles :) Let's see when (and if!) TensorFlow alone won't be enough.

If you missed the articles about the previous days' solutions, here's a handy list:

- [Day 1](/tensorflow/2021/12/11/advent-of-code-tensorflow/)
- [Day 2](/tensorflow/2021/12/12/advent-of-code-tensorflow-day-2/)
- [Day 3](/tensorflow/2021/12/14/advent-of-code-tensorflow-day-3/)
- [Day 4](/tensorflow/2021/12/17/advent-of-code-tensorflow-day-4/)
- [Day 5](/tensorflow/2021/12/22/advent-of-code-tensorflow-day-5/)
- [Day 6](/tensorflow/2021/12/25/advent-of-code-tensorflow-day-6/)

The next article will be about my solution to [Day 8](https://adventofcode.com/2021/day/8) problem. I'd be honest, that solution is ugly since the problem itself requires a bunch of `if` statement for solving it - nothing exciting. The solution to Day 9 problem, instead, is way more interesting because I solved it using lots of computer vision concepts like image gradients and flood fill algorithm!

For any feedback or comment, please use the Disqus form below - thanks!
