---
layout: post
title: "Advent of Code 2021 in pure TensorFlow - day 4"
date: 2021-12-17 08:00:00
categories: tensorflow
summary: "Using tensors for representing and manipulating data is very convenient. This representation allows changing shape, organizing, and applying generic transformations to the data. TensorFlow - by design - executes all the data manipulation in parallel whenever possible. The day 4 challenge is a nice showcase of how choosing the correct data representation can easily simplify a problem."
authors:
    - pgaleone
---

Using tensors for representing and manipulating data is very convenient. This representation allows changing shape, organizing, and applying generic transformations to the data.
TensorFlow - by design - executes all the data manipulation in parallel whenever possible. The day 4 challenge is a nice showcase of how choosing the correct data representation can easily simplify a problem.

The challenge itself is not complicated, but similarly to [day 3](/tensorflow/2021/12/14/advent-of-code-tensorflow-day-3/), we'll need to use a very dynamic element offered by the framework, the `TensorArray` data structure.

## [Day 4: Giant Squid](https://adventofcode.com/2021/day/4): part one

You can click on the title above to read the full text of the puzzle. The TLDR version is:

Let's play [bingo](https://en.wikipedia.org/wiki/Bingo_(American_version))! Our puzzle input is a text file containing in the first line a comma separated list of drawn numbers, like

```
7,4,9,5,11,17,23,2,0,14,21,24,10,16,13,6,15,25,12,22,18,20,8,19,3,26,1
```

While the rest of the file contains the `5x5` boards. Each board is a made of 5 **rows** and 5 **columns** and each board is separated with an empty line.

```
22 13 17 11  0
 8  2 23  4 24
21  9 14 16  7
 6 10  3 18  5
 1 12 20 15 19

 3 15  0  2 22
 9 18 13 17  5
19  8  7 25 23
20 11 10 24  4
14 21 16 12  6

14 21 17 24  4
10 16 15  9 19
18  8 23 26 20
22 11 13  6  5
 2  0 12  3  7
```

The challenge consists in simulating the game. We need, thus, to keep track for every board of the drawn numbers. A player wins the game when one of its board has at least one **complete row or column** of marked numbers.

The text asks to calculate the **score** of the winning board as **the sum of all the unmarked numbers** on that board, multiplied by **the number that was just called**.

In the example, the last one is the winning board, and **bingo** has been made after drawing 7,4,9,5,11,17,23,2,0,14,21, and 24. Thus the sum of all the **unmarked** numbers is `188` and the final score is `188 * 24 = 4512`.

### Design phase

We need to simulate the complete game. Therefore we need some good representation of the boards that should allow us to easily find rows and columns with marked numbers. Representing every board as a `tf.Tensor` with `shape=(5,5)` can be the best choice.

We also need to find a way to **keep track** for every board of the drawn numbers. The TensorArray data structure, as shown in [Day 3](/tensorflow/2021/12/14/advent-of-code-tensorflow-day-3/), is the perfect fit. This data structure allows us to store arbitrary-shaped `tf.Tensor` objects, thus we can easily create an array of `5x5` boards.

We are **not** interested in the value of the marked numbers, hence we can overwrite them with some value that will never be drawn (-1). After every extraction, we should just check if there's a winner board, for stopping the execution and compute the score of the winning board.

The data representation we use is perhaps the most important part of the solution, and it's all created in the data pipeline.

### Input pipeline

We create a `tf.data.Dataset` object for reading the text file line-by-line [as usual](/tensorflow/2021/12/11/advent-of-code-tensorflow/#input-pipeline). But this time, since the data is heterogeneous, we create two different datasets. One dataset produces the drawn numbers, and the other produces the boards.

```python
dataset = tf.data.TextLineDataset("input")

# The first row is a csv line containing the number extracted in sequence
extractions = (
    dataset.take(1)
    .map(lambda line: tf.strings.split(line, ","))
    .map(lambda strings: tf.strings.to_number(strings, out_type=tf.int64))
    .unbatch()
)

# All the other rows are the boards, every 5 lines containing an input
# is a board. We can organize the boards as elements of the dataset, a dataset of boards
boards = (
    dataset.skip(1)
    .filter(lambda line: tf.greater(tf.strings.length(line), 0))
    .map(tf.strings.split)
    .map(
        lambda string: tf.strings.to_number(string, out_type=tf.int64)
    )  # row with 5 numbers
    .batch(5)  # board 5 rows, 5 columns
)
```

`extractions` is a dataset that produces a scalar on every iteration. The `unbatch` method is used to unpack a single `tf.Tensor` containing all the numbers to a sequence of scalars.
It's worth noting that `tf.strings.to_number` is applied to a Tensor containing all the numbers in `tf.string` format, hence this application is executed in parallel on all the values.

`boards` is a dataset created skipping the first line (previously used), and by removing all the empty lines. The usual string to integer conversion is applied and we conclude by calling `batch(5)`. Since every read line contains `5` elements, by batching them together we end up with a dataset that produces a tensor with a `5x5` shape on every iteration.

### Computing the score

In the design phase, we decided to put a `-1` on the board where there's a match. Hence, we can easily define a helper function that computes and prints the score of a board.

```python
def _score(board, number):
    tf.print("Winner board: ", board)
    tf.print("Last number: ", number)

    # Sum all unmarked numbers
    unmarked_sum = tf.reduce_sum(
        tf.gather_nd(board, tf.where(tf.not_equal(board, -1)))
    )
    tf.print("Unmarked sum: ", unmarked_sum)

    final_score = unmarked_sum * number
    tf.print("Final score: ", final_score)
```

The function requires the board to use and the last drawn number to compute the final score. Summing all the unmarked numbers is trivial, in fact, it's just a matter of filtering for all the values different from `-1`, gathering the values, and passing them to `tf.reduce_sum` that when used without specifying the reduction dimension will sum up all the values in the input tensor producing a scalar value.

The final score is then easily computed.

### Playing bingo

Our TensorFlow program needs a state. In particular, we need to know when to stop looping over the extractions (a boolean state), save the last drawn number, and also the winner board. These states should also be returned by our TensorFlow program, so to use the `_score` function previously defined.

Moreover, we need a `TensorArray` object for storing in a mutable data structure the various boards. I want to stress out that the `TensorArray` is one of the few **totally mutable objects** TensorFlow provides. Differently from `tf.Variable` the elements of a `tf.TensorArray` can change shape, and the array can grow past its original size.

```python
class Bingo(tf.Module):
    def __init__(self):
        # Assign every board in a TensorArray so we can read/write every board
        self._ta = tf.TensorArray(dtype=tf.int64, size=1, dynamic_size=True)

        self._stop = tf.Variable(False, trainable=False)

        self._winner_board = tf.Variable(
            tf.zeros((5, 5), dtype=tf.int64), trainable=False
        )
        self._last_number = tf.Variable(0, trainable=False, dtype=tf.int64)
```

We need to decide when to stop the extraction loop, for doing so it will be useful to design an `is_winner` function that given a board checks if it has a row or a column full with marked objects.

During the design phase, we decided to apply `-1` in the marked position, hence the `is_winner` function can be easily defined as follows.

```python
@staticmethod
def is_winner(board: tf.Tensor) -> tf.Tensor:
    rows = tf.reduce_sum(board, axis=0)
    cols = tf.reduce_sum(board, axis=1)

    return tf.logical_or(
        tf.reduce_any(tf.equal(rows, -5)), tf.reduce_any(tf.equal(cols, -5))
    )
```

The `axis` parameter of `tf.reduce_sum` defines the reduction dimension to use. In practice, we are summing along the rows and the column (respectively in the two calls) getting two tensors with shape `(5)` containing the sum over these dimensions. If we marked a row/column, then one of these 5 values will be a `-5`.

Differently from what we made for solving the day 4 puzzle, where we only used the `stack` and `unstack` method of `tf.TensorArray`, this time we use only `unstack` to populate the `TensorArray` with the boards, and then we used `read` and `write` methods to read/write singularly every board.

```python
# @tf.function
def __call__(
    self, extractions: tf.data.Dataset, boards: tf.data.Dataset
) -> Tuple[tf.Tensor, tf.Tensor]:
    # Convert the datasaet to a tensor  and assign it to the ta
    # use the tensor to get its shape and know the number of boards
    tensor_boards = tf.convert_to_tensor(list(boards))  # pun intended
    tot_boards = tf.shape(tensor_boards)[0]
    self._ta = self._ta.unstack(tensor_boards)

    # Remove the number from the board when extracted
    # The removal is just the set of the number to -1
    # When a row or a column becomes a line of -1s then bingo!
    for number in extractions:
        if self._stop:
            break
        for idx in tf.range(tot_boards):
            board = self._ta.read(idx)
            board = tf.where(tf.equal(number, board), -1, board)
            if self.is_winner(board):
                self._stop.assign(tf.constant(True))
                self._winner_board.assign(board)
                self._last_number.assign(number)
                break
            self._ta = self._ta.write(idx, board)
    return self._winner_board, self._last_number
```

Note that `write` produces an "operation" that must be assigned to itself (`self._ta = self._ta.write(idx, board)`).

### Execution

We have all we need to play the game and compute the final result

```python
bingo = Bingo()
winner_board, last_number = bingo(extractions, boards)
_score(winner_board, last_number)
```

Here we go, part 1 solved! We are ready for part 2.

## [Day 4: Giant Squid](https://adventofcode.com/2021/day/4): part two

Part 2 requires to **figure out which board will win last** and compute, thus, the final score of this board.

### Design phase - part two

Finding the last winning board requires drawing all the numbers, playing the game, ignore the winning boards until we reach the end of the loop.
The idea is to remove from the game a board as soon as it becomes a winning board. In this way, we don't place `-1` where's no more needed (otherwise we end up with all the boards full of `-1`).

Removing a board from the game is trivial thanks to our representation. Whenever we find a winning board (row/column full of -1) we can just set **all the values** of this board to a number that will never be drawn: `0`.

### Invalidating the boards

With a small modification of the `call` method, we can solve the puzzle.

```python
def __call__(
    self,
    extractions: tf.data.Dataset,
    boards: tf.data.Dataset,
    first_winner: tf.Tensor = tf.constant(True),
) -> Tuple[tf.Tensor, tf.Tensor]:
    # Convert the datasaet to a tensor  and assign it to the ta
    # use the tensor to get its shape and know the numnber of boards
    tensor_boards = tf.convert_to_tensor(list(boards))  # pun intended
    tot_boards = tf.shape(tensor_boards)[0]
    self._ta = self._ta.unstack(tensor_boards)

    # Remove the number from the board when extracted
    # The removal is just the set of the number to -1
    # When a row or a column becomes a line of -1s then bingo!
    for number in extractions:
        if self._stop:
            break
        for idx in tf.range(tot_boards):
            board = self._ta.read(idx)
            board = tf.where(tf.equal(number, board), -1, board)
            if self.is_winner(board):
                self._winner_board.assign(board)
                self._last_number.assign(number)
                if first_winner:
                    self._stop.assign(tf.constant(True))
                    break
                # When searching for the last winner
                # we just invalidate every winning board
                # by setting all the values to zero
                board = tf.zeros_like(board)
            self._ta = self._ta.write(idx, board)
    return self._winner_board, self._last_number
```

We added the `first_winner` parameter that will change the behavior of the `call` method: when `True` it behaves like required for part 1, when false it finds the last winning board.

### Execution - part two

Very easy

```python
## --- Part Two ---
# Figure out the last board that will win
bingo = Bingo()
last_winner_board, last_number = bingo(extractions, boards, tf.constant(False))
_score(last_winner_board, last_number)
```

It works! Problem 4 is solved!

## Conclusion

You can see the complete solution in folder `4` on the dedicated Github repository: [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

We are using `tf.TensorArray` declaring them in the `__init__` and this leads to the [very same limitations](/tensorflow/2021/12/14/advent-of-code-tensorflow-day-3/#tensorarray-limitation-in-graph-mode) we faced while solving the [day 3 puzzle](/tensorflow/2021/12/14/advent-of-code-tensorflow-day-3/).

I already solved puzzles 5 and 6 and both have been fun. In particular, while solving the day 6 puzzle I realized that the limitation I discovered only depends on where we declare and use TensorArray. My solution for that puzzle uses a `tf.TensorArray` in a `tf.function`-decorated function and it works pretty well! So stay tuned for that article!

The next article, however, will be about my solution to the day 5 puzzle, which has been solved without using any `tf.TensorArray` but contains some easy but interesting mathematical concept that's worth writing about :)

For any feedback or comment, please use the Disqus form below - thanks!
