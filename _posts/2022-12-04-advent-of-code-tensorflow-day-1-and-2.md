---
layout: post
title: "Advent of Code 2022 in pure TensorFlow - Days 1 & 2"
date: 2022-12-04 08:00:00
categories: tensorflow
summary: "Let's start a tradition. This is the second year in a row I try to solve the Advent of Code (AoC) puzzles using only TensorFlow. This article contains the description of the solutions of the Advent of Code puzzles 1 and 2, in pure TensorFlow."
authors:
    - pgaleone
---

Let's start a tradition. This is the second year in a row I try to solve the [Advent of Code](https://adventofcode.com/) (AoC) puzzles using <b>only</b> TensorFlow.

[Last year](/tensorflow/2021/12/11/advent-of-code-tensorflow/) I found it very interesting to write TensorFlow programs (that's <b>NOT</b> machine learning stuff!) for solving this programming challenge. I discovered several peculiarities and not widely used features of TensorFlow - other than bugs/limitations of the framework itself! - and this year I'm ready to do the same.

Why use TensorFlow? Three simple reasons

1. I like the framework and I'm a huge fan of the [SavedModel file format](https://www.tensorflow.org/guide/saved_model). I find extremely powerful the idea of describing the computation using a graph and having a language-agnostic representation of the computation that I can reuse everywhere TensorFlow can run (e.g. all the bindings created through the language C FFI and the TensorFlow C library, and all the specialized runtimes like TensorFlow Lite, TensorFlow JS, ...
1. Solving coding puzzles in this way is a great way for building fluency.
1. It's fun!

This year I kinda-started on time. Last year I started when the competition was already halfway through. This year instead, I'll try to solve every puzzle the very same day it's published.

One year ago, I wrote an article for every single problem I solved (here's [the wrap-up of Advent of Code 2021 in pure TensorFlow](/tensorflow/2022/01/21/advent-of-code-tensorflow-wrap-up/). Let's see where this year brings us. I could write an article per solution, or group them together. I don't know it yet. I guess it depends on how complex a solution is and if it's worth writing an article for something "boring". For this reason, I described in this first article the solution to the first **and second** problems, because I found the latter boring and writing an article about this solution alone would be a waste.
        
Of course, I'd try to highlight the TensorFlow peculiarity every time I use something not widely used or whenever I discover something noticeable about the framework.

Let's start!

## [Day 1: Calorie Counting](https://adventofcode.com/2022/day/1): part one

You can click on the title above to read the full text of the puzzle. The TLDR version is:

The puzzle input is something like
```
1000
2000
3000

4000

5000
6000

7000
8000
9000

10000
```

where every group of numbers represents the number of Calories carried by an Elf. The first Elf is carrying 1000+2000+3000 = 6000 Calories, the second Elf 4000, and so on.

Here's the challenge:

> Find the Elf carrying the most Calories. How many total Calories is that Elf carrying?

In the example, this Elf is the fourth Elf with a Calories count of 24000.

### Design Phase

The problem can be breakdown into 4 simple steps:

1. Read the data
2. Identify the groups
3. Sum the calories in every group
4. Find the group containing the maximum amount of calories

### Input pipeline

The input pipeline is always the same for all the puzzles of the Advent of Code: the input is just a text file containing the puzzle input. Thus, we can read the `input_path` file using the `TextLineDataset` object. This `tf.data.Dataset` specialization automatically creates a new element for every new line in the file.

```python
def main(input_path: Path) -> int:
    dataset = tf.data.TextLineDataset(input_path.as_posix())
```

The dataset creation will always be the same, thus this part will not be presented in the solutions of the other puzzles presented in this article or (I guess) in other articles where the input requires no changes.

#### Identifying groups and summing their values: scan & filter

Identifying the groups requires us to check if the input line we are reading is empty: this is possible through the usage of the `tf.strings` package.

For accumulating the values, instead, there's no need to use a `tf.Variable`. The reading process is sequential and we can just use a state that's carried over the loop iteration.

[tf.data.Dataset.scan](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#scan)`(initial_state, scan_func)` is the perfect tool for this workflow. As the documentation states

> [tf.data.Dataset.scan is ] A transformation that scans a function across an input dataset.
> 
> This transformation is a stateful relative of `tf.data.Dataset.map`. In addition to mapping `scan_func` across the elements of the input dataset, `scan()` accumulates one or more state tensors, whose initial values are `initial_state`.

The `scan_func` maps the pair `(old_state, input_element)` to the pair `(new_state, output_element)`.

Since we'll use the condition on the empty line to identify the groups, we must add an empty line at the end of our dataset otherwise we won't be able to identify the last group in the input file.

```python
dataset = dataset.concatenate(tf.data.Dataset.from_tensors([""]))
initial_state = tf.constant(0, dtype=tf.int64)

@tf.function
def scan_func(state, line):
    if tf.strings.length(line) > 0:
        new_state = state + tf.strings.to_number(line, tf.int64)
        output_element = tf.constant(-1, tf.int64)
    else:
        new_state = tf.constant(0, tf.int64)
        output_element = state
    return new_state, output_element

dataset = dataset.scan(initial_state, scan_func)
```

The only thing worth noting is the usage of `tf.function`, which explicitly makes the reader understand that the `scan_func` function will be executed in a static-graph context.

Anyway, this is just style, since **every transformation applied using tf.data.Dataset methods is always executed in a static graph context**. Thus, if the `@tf.function` decoration wasn't present, nothing would have changed in the behavior, since it's `tf.data.Dataset` that converts to a static graph every transformation (for performance reasons).

The `dataset` object now contains something like

```
[-1, -1, -1, 6000, -1, 4000, -1, -1, 11000, -1, -1, -1, 24000, -1, 10000]
```

The `scan_func` used the constant `-1` as output element while it was processing the values in a group. Thus, for getting only the total amount of calories in the dataset we should filter it.

```python
dataset = dataset.filter(lambda x: x > 0)
```

#### Finding the maximum group

Unfortunately, it's not possible to use the `tf.reduce_*` functions over a dataset. Thus we need to convert the whole dataset to a `tf.Tensor` and than find the maximum value.

```python
tensor = tf.convert_to_tensor(list(dataset.as_numpy_iterator()))

max_calories = tf.reduce_max(tensor)
elf_id = tf.argmax(tensor) + 1
tf.print("## top elf ##")
tf.print("max calories: ", max_calories)
tf.print("elf id: ", elf_id)
```

Part one: ✅

## [Day 1: Calorie Counting](https://adventofcode.com/2022/day/1): part two

The problem is trivial: instead of finding only the Elf that carries the most Calories, the puzzle asks us to find the top 3 Elves carrying the most Calories and sum them.

In TensorFlow this is extremely trivial since the ranking operation is very common in Machine Learning (and not only in this domain), thus there's a function ready to use: [tf.math.top_k](https://www.tensorflow.org/api_docs/python/tf/math/top_k?hl=en).

```python
tf.print("## top 3 elves ##")
top_calories, top_indices = tf.math.top_k(tensor, k=3)
tf.print("calories: ", top_calories)
tf.print("indices: ", top_indices + 1)
tf.print("sum top calories: ", tf.reduce_sum(top_calories))
```

The first puzzle is completely solved! Let's go straight to the second challenge.

## [Day 2: Rock Paper Scissors](https://adventofcode.com/2022/day/2): part one

You can click on the title above to read the full text of the puzzle. The TLDR version is:

**Disclaimer**: I found this problem boring. So I won't go into the detail of the solution a lot, since the algorithmic part is just playing Rock Paper Scissors 😂

The puzzle input is something like

```
A Y
B X
C Z
```

where every line represents a round of [Rock paper scissors](https://en.wikipedia.org/wiki/Rock_paper_scissors). The first element of each line represents the choice of the opponent, while the second element it's our move.

The total score is computed using this rule: the outcome + the shape we selected.

This is the mapping:

- A: Rock.
- B: Paper.
- C: Scissors.
- X: Rock. Value 1.
- Y: Paper. Value 2.
- Z: Scissor. Value 3.

The outcomes instead have these values:

- Lost. Value: 0.
- Draw. Value: 3.
- Won. Value: 6.

The puzzle asks us to compute the total score of the matches. Given the sample input, the total score will be 8+1+6=15.

### Ragged Tensors & Mappings

The input dataset can be easily split using [tf.strings.split](https://www.tensorflow.org/api_docs/python/tf/strings/split?hl=en) that given a string `tf.Tensor` of rank `N` returns a `RaggedTensor` of rank `N+1`. When working with strings, [RaggedTensors](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor?hl=en) are of fundamental importance since they allow us to work with tensors with one or more dimensions whose slices may have different lengths.

```python
opponent_action = dataset.map(lambda line: tf.strings.split(line, " "))
```

The `opponent_action` is a `tf.data.Dataset` of ragged tensors. It can be seen as a dataset of pairs (since our lines only contain 2 strings separated by a space), where the opponent is in position `0` and our action is in position `1`.

Since the final score requires us to map our action to its value, we can use a [`tf.lookup.StaticHashTable`](https://www.tensorflow.org/api_docs/python/tf/lookup/StaticHashTable) that perfectly satisfies our needs. In fact, this is an immutable map.

```python
keys_tensor = tf.constant(["X", "Y", "Z"])
vals_tensor = tf.constant([1, 2, 3])

action_to_score = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
    default_value=-1,
)
```

### Play the game

As I anticipated, I won't explain this part since it's really just playing the game and returning the score of the `outcome` together with the score assigned with our action.

```python
@tf.function
def play(opponent_action):
    opponent = opponent_action[0]
    action = opponent_action[1]
    outcome = 3
    my_action_score = action_to_score.lookup(action)
    if tf.equal(opponent, "A"):
        if tf.equal(action, "Y"):
            outcome = 6
        if tf.equal(action, "Z"):
            outcome = 0
    if tf.equal(opponent, "B"):
        if tf.equal(action, "X"):
            outcome = 0
        if tf.equal(action, "Z"):
            outcome = 6
    if tf.equal(opponent, "C"):
        if tf.equal(action, "X"):
            outcome = 6
        if tf.equal(action, "Y"):
            outcome = 0
    return outcome + my_action_score

opponent_action_played = opponent_action.map(play)

tf.print(
    "sum of scores according to strategy: ",
    tf.reduce_sum(
        tf.convert_to_tensor(list(opponent_action_played.as_numpy_iterator()))
    ),
)
```

Part one gone.

## [Day 2: Rock Paper Scissors](https://adventofcode.com/2022/day/2): part two

Part two requires us to change the interpretation of the dataset. Instead of considering XYZ our moves, we should consider them the desired outcome for every match. This is the mapping to consider:

- X: Lose.
- Y: Draw.
- Z: Win.

The problem doesn't change. It requires a new mapping ad a new play function, in which we play the game knowing the outcome.

```python
outcome_to_score = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(["X", "Y", "Z"]), tf.constant([0, 3, 6])
    ),
    default_value=-1,
)

@tf.function
def play_knowing_outcome(opponent_outcome):
    opponent = opponent_outcome[0]
    outcome = opponent_outcome[1]

    # draw
    my_action = tf.constant("Z")
    if tf.equal(outcome, "Y"):
        if tf.equal(opponent, "A"):
            my_action = tf.constant("X")
        if tf.equal(opponent, "B"):
            my_action = tf.constant("Y")
    # lose
    if tf.equal(outcome, "X"):
        if tf.equal(opponent, "A"):
            my_action = tf.constant("Z")
        if tf.equal(opponent, "B"):
            my_action = tf.constant("X")
        if tf.equal(opponent, "C"):
            my_action = tf.constant("Y")

    # win
    if tf.equal(outcome, "Z"):
        if tf.equal(opponent, "A"):
            my_action = tf.constant("Y")
        if tf.equal(opponent, "B"):
            my_action = tf.constant("Z")
        if tf.equal(opponent, "C"):
            my_action = tf.constant("X")

    return action_to_score.lookup(my_action) + outcome_to_score.lookup(outcome)

opponent_outcome = opponent_action
opponent_outcome_played = opponent_outcome.map(play_knowing_outcome)

tf.print(
    "sum of scores according to new strategy: ",
    tf.reduce_sum(
        tf.convert_to_tensor(list(opponent_outcome_played.as_numpy_iterator()))
    ),
)
```

The algorithmic solution is boring so it's not worth analyzing it. Anyway, this algorithm that's completely executed as a static-graph representation looping over the elements of a `tf.data.Dataset` allowed us to solve the problem!

## Conclusion

For the second year in a row, I'm trying to solve the AoC puzzles in pure TensorFlow.

The [last year](/tensorflow/2022/01/21/advent-of-code-tensorflow-wrap-up/) has been fun, although I started late and for this reason I completed and described only half of the puzzles.
Perhaps this year I will find the time to complete and describe - with a series of articles - more solutions.

The goal of this article series is to demonstrate how TensorFlow can be used as a "general purpose" programming language and, at the same time, talk about not widely used TensorFlow's features - if the puzzles require them!

In general, in every article I will try to explain how to reason when writing TensorFlow programs, always stressing the static-graph representation. Even in solving these first two challenges we already encountered:

- Ragged tensors.
- The dataset objects and the transformations that are always executed in graph mode.
- The lookup tables (that last year were inside the experimental package).

Let's see where the next challenges will bring us!

For any feedback or comment, please use the Disqus form below - thanks!

A last note. All the solutions are on Github. You can browse them (year by year, day by day): [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).
