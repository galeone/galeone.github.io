---
layout: post
title: "Advent of Code 2022 in pure TensorFlow - Day 10"
date: 2023-03-25 08:00:00
categories: tensorflow
summary: "Solving problem 10 of the AoC 2022 in pure TensorFlow is an interesting challenge. This problem involves simulating a clock signal with varying frequencies and tracking the state of a signal-strength variable. TensorFlow's ability to handle complex data manipulations, control structures, and its @tf.function decorator for efficient execution makes it a fitting choice for tackling this problem. By utilizing TensorFlow's features such as Dataset transformations, efficient filtering, and tensor operations, we can create a clean and efficient solution to this intriguing puzzle."
authors:
    - pgaleone
    - chatGPT
---

Solving problem 10 of the AoC 2022 in pure TensorFlow is an interesting challenge. This problem involves simulating a clock signal with varying frequencies and tracking the state of a signal-strength variable. TensorFlow's ability to handle complex data manipulations, control structures, and its `@tf.function` decorator for efficient execution makes it a fitting choice for tackling this problem. By utilizing TensorFlow's features such as Dataset transformations, efficient filtering, and tensor operations, we can create a clean and efficient solution to this intriguing puzzle.

## [Day 10: Clock Signal](https://adventofcode.com/2022/day/10)

You can click on the title above to read the full text of the puzzle. The TLDR version is: the puzzle involves a series of instructions to update a clock signal's strength. Each cycle, the clock signal's strength X is updated based on a given list of instructions. The goal is to calculate the sum of the signal strength at specific cycles and visualize the clock signal's behavior over a fixed number of cycles.

### Parsing the input

First, let's use `tf.data.TextLineDataset` to read the input file line by line:

```python
dataset = tf.data.TextLineDataset(input_path.as_posix())
```

Now, split each line into a list of strings (the operation and the value):

```python
dataset = dataset.map(lambda line: tf.strings.split(line, " "))
```

Then, we need to define a function `opval` to convert the string values into a tuple of `(op, val)`. If the operation is `"noop"`, the value will be set to `0`.

```python
@tf.function
def opval(pair):
    if tf.equal(tf.shape(pair)[0], 1):
        return pair[0], tf.constant(0, tf.int32)

    return pair[0], tf.strings.to_number(pair[1], tf.int32)

dataset = dataset.map(opval)
```

As usual, when working with a `tf.data.Dataset` the eager mode is disabled and everything runs in graph mode. That's why we explicitly added the `tf.function` decorator on top of the `opval` function (although not required - but it helps to remember that we need to think in graph mode).

We'll use a lookup table (`tf.lookup.StaticHashTable`) to map instruction strings to integer values. This allows us to work with numerical values, which is more convenient for processing in TensorFlow.

```python
lut = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(["noop", "addx"]), tf.constant([0, 1])
    ),
    default_value=-1,
)
```

Next, we need to process the dataset so that each element represents a clock cycle. To do this, we define a `prepend_noop` function that inserts a `"noop"` instruction before each `"addx"` instruction. This will ensure that the dataset correctly represents the clock signal's behavior.

```python
@tf.function
def prepend_noop(op, val):
    if tf.equal(op, "noop"):
        return tf.stack([noop, invalid], axis=0)

    return tf.stack(
        [
            noop,
            tf.stack((lut.lookup(tf.expand_dims(op, axis=0))[0], val), axis=0),
        ],
        axis=0,
    )

dataset = (
    dataset.map(prepend_noop)
    .unbatch()
    .filter(lambda op_val: tf.not_equal(op_val[0], -1))  # remove invalid
    .map(lambda op_val: (op_val[0], op_val[1]))
)
```

Now that we have the dataset correctly formatted, we can proceed with simulating the clock signal's behavior.

### Simulating the clock signal

We'll use a TensorFlow `tf.Variable` to keep track of the current cycle and the current signal strength X. Initialize these variables as follows:

```python
cycle = tf.Variable(0, dtype=tf.int32)
X = tf.Variable(1, dtype=tf.int32)
```

To simulate the clock signal's behavior, we define a `clock` function that processes each instruction in the dataset. This function updates the cycle and signal strength `X` variables accordingly.

```python
prev_x = tf.Variable(X)

def clock(op, val):
    prev_x.assign(X)
    if tf.equal(op, noop_id):
        pass
    else:  # addx
        X.assign_add(val)

    cycle.assign_add(1)

    if tf.reduce_any([tf.equal(cycle, value) for value in range(20, 221, 40)]):
        return [cycle, prev_x, prev_x * cycle]
    return [cycle, prev_x, -1]
```

Next, we'll create a dataset of signal strength values at the specific cycles requested in the problem (i.e., every 40 cycles between 20 and 220 inclusive). We do this by mapping the clock function to the dataset and filtering out the elements with a signal strength value of -1.

```python
strenghts_dataset = dataset.map(clock).filter(
    lambda c, x, strenght: tf.not_equal(strenght, -1)
)

strenghts = tf.convert_to_tensor((list(strenghts_dataset.as_numpy_iterator())))
```

Now, we can calculate the sum of the six signal strength values:

```python
sumsix = tf.reduce_sum(strenghts[:, -1])
tf.print("Sum of six signal strenght: ", sumsix)
```

In the provided solution, we used the `@tf.function` decorator in the `opval` and `prepend_noop` methods. This powerful feature of TensorFlow enables automatic conversion of a Python function into a TensorFlow graph. The main benefits of this conversion are performance improvements and better compatibility with TensorFlow operations.

By converting a function into a TensorFlow graph, we allow TensorFlow to optimize the computation by fusing operations and running them more efficiently. This can lead to significant speed improvements, especially for functions that are called repeatedly, as in our case when processing the input dataset.

Part one solved!

### Visualizing the clock signal

The second part of the puzzle asks us to visualize the clock signal's behavior over a fixed number of cycles. For doing it we'll create a `tf.Variable` to store the clock signal visualization in a 2D grid. Initialize this variable as follows:

```python
crt = tf.Variable(tf.zeros((6, 40, 1), tf.string))
```

Next, we'll define a `clock2` function to update the visualization grid based on the clock signal's behavior. This function modifies the grid at each cycle according to the current signal strength.

```python
row = tf.Variable(0, dtype=tf.int32)

def clock2(op, val):
    prev_x.assign(X)
    if tf.equal(op, noop_id):
        pass
    else:  # addx
        X.assign_add(val)

    modcycle = tf.math.mod(cycle, 40)
    if tf.reduce_any(
        [
            tf.equal(modcycle, prev_x),
            tf.equal(modcycle, prev_x - 1),
            tf.equal(modcycle, prev_x + 1),
        ]
    ):
        crt.assign(
            tf.tensor_scatter_nd_update(
                crt, [[row, tf.math.mod(cycle, 40)]], [["#"]]
            )
        )
    else:
        crt.assign(
            tf.tensor_scatter_nd_update(
                crt, [[row, tf.math.mod(cycle, 40)]], [["."]]
            )
        )

    cycle.assign_add(1)

    if tf.equal(tf.math.mod(cycle, 40), 0):
        row.assign_add(1)
    return ""
```

Finally, we map the `clock2` function to the dataset, and then we can print the resulting visualization:

```python
list(dataset.map(clock2).as_numpy_iterator())

tf.print(tf.squeeze(crt), summarize=-1)
```

Sqeezing the unary dimensions is necessary to correctly display the 2D grid without square brackets everywhere. The `summarize=-1` paramete of `tf.print` disables the standard behavior of printing only some part of the content and adding `...` in between. In this way, we can see directly in the terminal the letters.

Part 2 solved!

## Conclusion

You can the solution in folder `10` in the dedicated GitHub repository (in the `2022` folder): [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

This TensorFlow-based solution demonstrates the power and flexibility of the TensorFlow library, allowing us to efficiently solve the AoC 2022 problem 10. We used various TensorFlow operations, data structures, and functions to parse the input, simulate the clock signal's behavior, and visualize the clock signal's behavior over a fixed number of cycles. This approach showcases how TensorFlow can be utilized beyond its primary use case of deep learning, and can be employed to solve a wide range of computational problems.

By employing TensorFlow's built-in operations and data structures, we were able to efficiently process the input data, handle branching logic, and maintain state throughout the simulation. The final visualization provides a clear representation of the clock signal's behavior, and the sum of the six signal strengths is the solution to the problem.

As the AoC 2022 puzzles continue to challenge participants with new and diverse problems, this solution demonstrates that TensorFlow can be a powerful tool in the problem solver's toolkit. It serves as an example of how TensorFlow's flexibility extends beyond deep learning applications and can be effectively used to tackle complex problems in a variety of domains.


If you missed the article about the previous days’ solutions, here's a handy list

- [Advent of Code 2022 in pure TensorFlow - Days 1 & 2](/tensorflow/2022/12/04/advent-of-code-tensorflow-day-1-and-2/).
- [Advent of Code 2022 in pure TensorFlow - Days 3 & 4](/tensorflow/2022/12/11/advent-of-code-tensorflow-day-3-and-4/).
- [Advent of Code 2022 in pure TensorFlow - Day 5](/tensorflow/2022/12/21/advent-of-code-tensorflow-day-5/)
- [Advent of Code 2022 in pure TensorFlow - Day 6](/tensorflow/2022/12/27/advent-of-code-tensorflow-day-6/)
- [Advent of Code 2022 in pure TensorFlow - Day 7](/tensorflow/2022/12/29/advent-of-code-tensorflow-day-7/)
- [Advent of Code 2022 in pure TensorFlow - Day 8](/tensorflow/2023/01/14/advent-of-code-tensorflow-day-8/)
- [Advent of Code 2022 in pure TensorFlow - Day 9](/tensorflow/2023/01/23/advent-of-code-tensorflow-day-9/)

For any feedback or comment, please use the Disqus form below - thanks!
