---
layout: post
title: "Advent of Code 2022 in pure TensorFlow - Day 9"
date: 2023-01-23 08:00:00
categories: tensorflow
summary: "In this article, we'll show two different solutions to the Advent of Code 2022 day 9 problem. Both of them are purely TensorFlow solutions. The first one, more traditional, just implement a solution algorithm using only TensorFlow's primitive operations - of course, due to some TensorFlow limitations this solution will contain some details worth reading (e.g. using a pairing function for being able to use n-dimensional tf.Tensor as keys for a mutable hashmap). The second one, instead, demonstrates how a different interpretation of the problem paves the way to completely different solutions. In particular, this solution is Keras based and uses a multi-layer convolutional model for modeling the rope movements."
authors:
    - pgaleone
    - tjwei
---

In this article, we'll show two different solutions to the Advent of Code 2022 day 9 problem. Both of them are purely TensorFlow solutions. The first one, more traditional, just implement a solution algorithm using only TensorFlow's primitive operations - of course, due to some TensorFlow limitations this solution will contain some details worth reading (e.g. using a pairing function for being able to use n-dimensional `tf.Tensor` as keys for a mutable hashmap). The second one, instead, demonstrates how a different interpretation of the problem paves the way to completely different solutions. In particular, this solution is Keras based and uses a multi-layer convolutional model for modeling the rope movements.

## [Day 9: Rope Bridge](https://adventofcode.com/2022/day/9)

You can click on the title above to read the full text of the puzzle. The TLDR version is: we need to model a rope movement. In part 1, there are only 2 nodes in the rope: Head (H) and Tail (T). Part 2, makes the problem a little bit more complicated, asking us to model the movement of a rope made of 10 knots.

The rope moves following a series of motions (the puzzle input) and it always respects a simple rule: the head moves and the tail follows it. In the first part of the problem, where the rope is short and it's made of only 2 knots, the head and tail must always be touching (diagonally adjacent and even overlapping both counts as touching).

How to model the movement in the 2 knots scenario is perfectly explained in the [puzzle](https://adventofcode.com/2022/day/9). We just report the relevant parts below:

```
....
.TH.
....

....
.H..
..T.
....

...
.H. (H covers T)
...
```

If the head is ever two steps directly up, down, left, or right from the tail, the tail must also move one step in that direction so it remains close enough:

```
.....    .....    .....
.TH.. -> .T.H. -> ..TH.
.....    .....    .....
```
```
...    ...    ...
.T.    .T.    ...
.H. -> ... -> .T.
...    .H.    .H.
...    ...    ...
```

Otherwise, if the head and tail aren't touching and aren't in the same row or column, the tail always moves one step diagonally to keep up:

```
.....    .....    .....
.....    ..H..    ..H..
..H.. -> ..... -> ..T..
.T...    .T...    .....
.....    .....    .....
```
```
.....    .....    .....
.....    .....    .....
..H.. -> ...H. -> ..TH.
.T...    .T...    .....
.....    .....    .....
```

The puzzle asks us to work out where the tail goes as the head follows a series of motions, assuming the head and the tail both start at the same position (the origin), overlapping.

Thus, given an input like

```
R 4
U 4
L 3
D 1
R 4
D 1
L 5
R 2
```

we need to simulate the rope movement and can count up all of the positions the tail visited at least once. In this diagram, `s` marks the starting position (which the tail also visited), and `#` marks other positions the tail visited:

```
..##..
...##.
.####.
....#.
s###..
```

So, there are 13 positions the tail visited at least once (the puzzle answer).

### Imperative solution: prerequisites

We can solve both parts of the puzzle by the simple observation that a rope is made of a Head, a Tail, and a variable number of knots in between (part 1: 0, part 2: 8). In part 1, the tail always moves together with the head, but in part 2 we first need to move by 10 times before the tail starts moving.

This assumption allows us to define the problem structure in a more generic way. We'll define a function (`get_play(nodes)`) that will model the rope movement depending on the number of nodes of the rope.

Before doing it, of course, we need a way to understand if two elements are close in the 2D grid. As usual, TensorFlow comes with a function ready to use for us. : [`tf.norm`](https://www.tensorflow.org/api_docs/python/tf/norm) with its `ord` parameter set to numpy `inf` correctly implements the [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance): [`tf.norm`](https://www.tensorflow.org/api_docs/python/tf/norm), that's the perfect choice for measuring distances on a 2D grid.

```python
def are_neigh(a, b):
    return tf.math.less_equal(tf.norm(a - b, ord=tf.experimental.numpy.inf), 1)
```

This function returns a boolean tensor if the two tensors `a` and `b` are neighbors according to the L∞ metric.

Since we are interested in keeping track of the position of the tail, we should find a way for saving all the visited positions. In pure Python, we can use a `tuple` with the `(x,y)` coordinates of the visited point as index of a map, or as elements of a set (since tuples are hashable). In pure TensorFlow this is not possible, since `tf.Tensors` are not hashable.

[`tf.lookup.experimental.MutableHashTable`](https://www.tensorflow.org/api_docs/python/tf/lookup/experimental/MutableHashTable) can be used to store only rank-0 elements as keys (e.g. the scalar value `1` is a valid key, but the tuple `(1,2)` is not). Thus, to workaround this issue, we need a way to map a pair of numbers into a scalar value.

In this way, we can use a MutableHashMap to store the visited coordinates. The tool that perfectly solves this problem is a [Pairing function](Pairing function).

In particular, we implement the [Cantor pairing function](https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function). This function assigns one natural number to each pair of natural numbers, however, for storing coordinates it's not perfect. In fact, this function maps `(a,b)` to `c`, but it also maps `(b,a)` to `c`! Thus, we need to manually take care of this (we'll do it in the `get_play` function).

Moreover, this bijection works on natural numbers, but coordinates can be negative, thus we need also a way to map integers to naturals.

```python
# integers to naturals
def to_natural(z):
    if tf.greater_equal(z, 0):
        return tf.cast(2 * z, tf.int64)
    return tf.cast(-2 * z - 1, tf.int64)
```

This function maps an integer value `z` to a natural value. Thus, the container pairing function can be easily implemented by applying the definition found on Wikipedia.

```python
def pairing_fn(i, j):
    i, j = to_natural(i), to_natural(j)
    return (i + j) * (i + j + 1) // 2 + j
```

Alright, we now have something to start developing our solution.

### Imperative solution: input & play

Reading the input data is trivial. As usual, `tf.data.Dataset` simplifies this data transformation and allows us to obtain a `tf.RaggedTensor` containing a `tf.string` tensor with the direction, and a `tf.int64` tensor with the amount.

```python
dataset = (
    tf.data.TextLineDataset(input_path.as_posix())
    .map(lambda line: tf.strings.split(line, " "))
    .map(lambda pair: (pair[0], tf.strings.to_number(pair[1], tf.int64)))
)
```

In the previous section we prepared all we need to use a MutableHashTable using coordinates as key values, thus we can declare it

```python
pos = tf.lookup.experimental.MutableHashTable(tf.int64, tf.int64, (-1, 0, 0))
```

The `pos` MutableHashTable is used in this way:

1. Map the coordinates to a natural number
2. Check if this number is present as a key in the map.
3. If it's not present, insert the tuple `(1, x, y)` as value. `1` means the point with coordinate `(x,y)` have been visited once.
4. If it's present, check if the first value of the tuple is `1`. If not, set it to 2. In this case, we are handling the scenario in which we visited `x,y` and now we are visiting `y,x`.
5. The `x,y` coordinates are stored only for debugging purposes, but they are de facto unused in this solution.

Alright, we now know how to use the `pos` hashtable correctly. However, as we know from the 3 years ago article [Analyzing tf.function to discover AutoGraph strengths and subtleties](/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/) we can't declare a `tf.Variable` whenever we want while working in graph-mode because variables are special nodes in the graph (we want to define the variable once and use it). But we need a `tf.Variable` for modeling the rope. In fact, depending on the number of knots we should declare a different variable with a different shape, and this is not possible when working in graph mode.

For this reason, we define the `get_play` function as a configurator for the `play` function defined in its body (thus, we are defining and returning a closure). The `get_play` function scope defines a separate lexical environment we can use for creating a new `tf.Variable` whose lifetime is bounded with the lifetime of the closure returned. In short, every time we'll call `get_play(x)` a new `tf.Variable` is created because a new `tf.function`-decorated function is created (automatically by its usage in pure static-graph mode because of `tf.data.Dataset`).

```python
def get_play(nodes):
    rope = tf.Variable(tf.zeros((nodes, 2), tf.int64))
```

We can now define the `play` closure that implements the head-tail movement as described in the requirement.

```python
def play(direction, amount):

    sign = tf.constant(-1, tf.int64)
    if tf.logical_or(tf.equal(direction, "U"), tf.equal(direction, "R")):
        sign = tf.constant(1, tf.int64)

    axis = tf.constant((0, 1), tf.int64)
    if tf.logical_or(tf.equal(direction, "R"), tf.equal(direction, "L")):
        axis = tf.constant((1, 0), tf.int64)

    for _ in tf.range(amount):
        rope.assign(tf.tensor_scatter_nd_add(rope, [[0]], [sign * axis]))
        for i in tf.range(1, nodes):
            if tf.logical_not(are_neigh(rope[i - 1], rope[i])):
                distance = rope[i - 1] - rope[i]

                rope.assign(
                    tf.tensor_scatter_nd_add(
                        rope, [[i]], [tf.math.sign(distance)]
                    )
                )

                if tf.equal(i, nodes - 1):
                    mapped = pairing_fn(rope[i][0], rope[i][1])
                    info = pos.lookup([mapped])[0]
                    visited, first_coord, second_coord = (
                        info[0],
                        info[1],
                        info[2],
                    )
                    if tf.equal(visited, -1):
                        # first time visited
                        pos.insert(
                            [mapped],
                            [
                                tf.stack(
                                    [
                                        tf.constant(1, tf.int64),
                                        rope[i][0],
                                        rope[i][1],
                                    ]
                                )
                            ],
                        )

    return 0
```

I suggest to the readers to take their time to go through this snippet. Of course the `get_play` function should return the closure, so the body of our configuration function ends with

```python
return play
```

### Imperative solution: conclusion

Both parts can be solved by instantiating two different graphs (through `get_play(n)`) and looping over the dataset, executing step by step the rope movements.

```python
tf.print("Part 1: ")
pos.insert([pairing_fn(0, 0)], [(1, 0, 0)])
list(dataset.map(get_play(2)))
tail_positions = pos.export()[1]
visited_count = tf.reduce_sum(tail_positions[:, 0])
tf.print(visited_count)

tf.print("Part 2: ")
pos.remove(pos.export()[0])
pos.insert([pairing_fn(0, 0)], [(1, 0, 0)])
list(dataset.map(get_play(10)))
tail_positions = pos.export()[1]
visited_count = tf.reduce_sum(tail_positions[:, 0])
tf.print(visited_count)
```

So far so good, we've demonstrated how TensorFlow can be used (knowing how to use the paring functions to workaround, thus, some limitations) to solve this code challenge.

However, as anticipated, this article contains also a different solution implemented in a completely different way. In fact, the funny part about solving coding challenges is to see how a problem can be modeled differently, and how a different model can lead to a completely different (but still correct) solution!

### Keras & convolutional solution

An alternative way to solve this puzzle is to use a convolutional neural network to simulate the rope movement.
Observe that every non-head knot follows the knot in front of it and they will end up _touching_  each other (diagonally adjacent and even overlapping both counts as touching, i.e., maximum metric is at most 1), and when a knot moves, it turns out that the new position would be touching the original position as well. Thus, we have the following two observations:

- The movement of a non-head knot depends only on its position and the position of the knot.
- T non-head knot will never be too far away from the knot in front of it. the maximum metric will be at most 2.
In particular, we only need finitely many _local_ patterns to describe the movement.  For example, suppose we have two knots `H`  and  `T`. The following are a few patterns:

```
.H.
...
T..

.H.
...
.T.

.H.
.T.
...

..H
.T.
...

..H
...
T..
```

For each of these patterns, the tail will then be pulled to the center of the 3x3 grids.  In fact, these  are all of the patterns up to rotation and flipping.  All these patterns can be encoded into a few 3x3 convolution kernels.  Consider we are simulating on a 5x5 grid. The position of the head knot can be represented as a one-hot 5x5 array, e.g.
```
0 0 0 0 0
0 0 0 1 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
```


and the position of the tail may look something like the following:
```
0 0 0 0 0
0 0 0 0 0
0 1 0 0 0
0 0 0 0 0
0 0 0 0 0
```
These two arrays can be stacked into a 1x5x5x2 array (in channel last convention, the heading 1 is the batch size), which is the input of  our convolution layer.  Suppose we want to match the pattern:
```
...
..H
T..
```
Then we design a 3x3 convolution kernel with input channels=2 and output channels = 1.
The first part of the kernel looks like
```
0 0 0
0 0 1
0 0 0
```
which apples to the first channel, and the second part of the kernel looks like
```
0 0 0
0 0 0
1 0 0
```
which applies to the second channel.  The result looks like
```
0 0 0 0 0
0 0 2 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
```
which is the representation of the desired position of the tail in the next step.
If an incompatible pattern is applied in a similar manner, say
```
...
T.H
...
```
The result would look like
```
0 0 0 0 0
0 0 1 0 0
0 0 1 0 0
0 0 0 0 0
0 0 0 0 0
```
That is, only the correct pattern would output value $$2$$ at the desired position, all other positions would have value 0 or 1. By applying a bias value $$-1$$ (subtract 1 from every  output value), and applying the non-linear function ReLU, the output value $$2$$ becomes $$1$$ and all other values become $$0$$.
In principle, we can construct a convolution kernel for each pattern, and only one of them will match and all other patterns will output $$0$$. Then we simply sum up all the outputs, then we will get a one-hot representation of the new tail position. The sum-up operation can again be considered as a 2d convolution with 1x1 kernel. We name this layer _collect layer_ in contrast to the _move layers_ that match different movement patterns.
However, considering all rotations and flipping, there will be 25 patterns. We can further simplify the patterns. Consider the following 4 patterns:
```
.H.
...
T..

.H.
...
.T.

.H.
...
..T

.H.
.T.
...
```
Since there will be only one tail, we can combine the above 4 patterns into one pattern:
```
.H.
.T.
TTT
```
Therefore, we need only need 9 patterns. The original implementation groups 25 patterns into 9 patterns in a slightly different way.  Finally, we need to track which grid has been visited by the tail knot, we use another layer to keep the information and update the information in a similar manner.

The code can be illustrated as followings, where `L` is the length of the rope:
```python
# The following layers are for non-head knots movement
for i in range(1, L):
    # there are a 'move layer' and a 'collect layer' for each knot.
    move, collect = model.layers[i * 2 - 1:i * 2 + 1]
    W, b = move.get_weights()
    # First 1+L channels are unmodified.
    for t in range(1 + L):
        W[1, 1, t, t] = 2  # copy all, note that b=-1, so 1=2*1-1 unchanged.
    # The new position of knot j=i+1 depends on the current position of knot i, and knot j.
    j = i + 1  # knot j follows knot i
    # If knot i is adjacent to knot j(maximum distance<=1), then knot j stays the same position.
    W[:, :, i, 1 + L] = W[1, 1, j, 1 + L] = 1
    # the following kernels will match patterns like
    # X X X
    # _ _ _
    # _ i _
    # where knot j is at one of the X position and knot j is expected to moved to the center position.
    for n, k in enumerate([0, 2]):
        W[:, k, j, 1 + L + 1 + n] = W[1, 2 - k, i, 1 + L + 1 + n] = 1
        W[k, :, j, 1 + L + 3 + n] = W[2 - k, 1, i, 1 + L + 3 + n] = 1
    # the following kernels match the patterns like
    # j _ _
    # _ _ _
    # _ _ i
    # knot j is expected to moved to the center position.
    for n, (y, x) in enumerate(zip([0, 0, 2, 2], [0, 2, 0, 2])):
        W[y, x, j, 1 + L + 5 + n] = W[2 - y, 2 - x, i, 1 + L + 5 + n] = 1
    move.set_weights([W, b])
    # The collect layer collect the results matched by above patterns
    W, = collect.get_weights()
    # Copy the first 1+L channels, except channel j for knot j.
    for t in range(1 + L):
        W[..., t, t] = 1  # copy
    W[..., j, j] = 0
    # For channel j, sum up the last 9 channels. There will be exactly one position has value 1, and rest of the position are all 0.
    W[..., 1 + L:, j] = 1  # collect moves
    collect.set_weights([W])
# For the last layer, also collect the position of the tail. 0 represents and 1 represent unvisited.
# Because the non-linear function is relu, it will clip the negative values into 0.
W[..., 1 + L:, 0] = -1  # collect unvisited
collect.set_weights([W])
```
The head knot is moved in one of the up, down, left, right directions, according to the input data.
We use a 3x3 2d convolution similar to the above _move layer_ for each direction and rotate the kernel dynamically according to the input data. We simulate the rope movement on an `NxN` grid using the following code:
```python
# %% run the simulation
state = tf.zeros((1, N, N, 1 + L), dtype=tf.float32).numpy()
# Starts with  every knot at the center position
state[0, N // 2, N // 2, :] = 1
# Every position is marked as unvisited.
state[..., 0] = 1 - state[..., 0]
for n, line in enumerate(open('input.txt').read().splitlines()):
    tf.print(n, line)
    direction, num = line.split(' ')
    # Rotate the kernel of the first layer according to the direction.
    angle = {'R': 0, 'U': 1, 'L': 2, 'D': 3}[direction]
    head_move.set_weights([tf.transpose(tf.image.rot90(head_WT, angle), (1, 2, 3, 0))])
    # Simulate the movement num times
    for i in range(int(num)):
        state = model(state)
# Count visited positions.
print('Ans:', int(tf.reduce_sum(1 - state[..., 0])))
```

Rotating the kernel feels a bit like cheating. The network can be modified to take an additional conditional input on the direction. Furthermore, the whole network to modified into an [RNN cell layer](https://www.tensorflow.org/guide/keras/rnn#rnn_layers_and_rnn_cells) and the  simulation loop can be replaced by RNN inference on the sequence of directions.
We keep the code in its current form for explaining the process of transforming a local pattern-matching problem into a convolutional neural network. The above-mentioned modification, though fairly standard, is still very interesting as it is applied to solve a non-standard problem.

## Conclusion

You can see both solutions in folder `9` in the dedicated GitHub repository (in the `2022` folder): [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

This article demonstrated how to solve this coding challenge in pure TensorFlow in two completely different ways: the first one, that models the problem as a standard programming problem resolved using only TensorFlow primitives, and the second one instead models the problem completely differently and uses the properties of the convolutions for solving the problem in a very cool way!

If you missed the article about the previous days’ solutions, here's a handy list

- [Advent of Code 2022 in pure TensorFlow - Days 1 & 2](/tensorflow/2022/12/04/advent-of-code-tensorflow-day-1-and-2/).
- [Advent of Code 2022 in pure TensorFlow - Days 3 & 4](/tensorflow/2022/12/11/advent-of-code-tensorflow-day-3-and-4/).
- [Advent of Code 2022 in pure TensorFlow - Day 5](/tensorflow/2022/12/21/advent-of-code-tensorflow-day-5/)
- [Advent of Code 2022 in pure TensorFlow - Day 6](/tensorflow/2022/12/27/advent-of-code-tensorflow-day-6/)
- [Advent of Code 2022 in pure TensorFlow - Day 7](/tensorflow/2022/12/29/advent-of-code-tensorflow-day-7/)
- [Advent of Code 2022 in pure TensorFlow - Day 8](/tensorflow/2023/01/14/advent-of-code-tensorflow-day-8/)

For any feedback or comment, please use the Disqus form below - thanks!
