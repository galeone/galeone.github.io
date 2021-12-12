---
layout: post
title: "Advent of Code 2021 in pure TensorFlow - day 2"
date: 2021-12-12 08:00:00
categories: tensorflow
summary: "A Solution to the AoC day 2 puzzle in pure TensorFlow. How to use Enums in TensorFlow programs and the limitations of tf.Tensor used for type annotation"
authors:
    - pgaleone
---

Day 2 challenge is very similar to the one faced in [day 1](/tensorflow/2021/12/11/advent-of-code-tensorflow/). In this article, we'll see how to integrate Python enums with TensorFlow, while using the very same design nuances used for the day 1 challenge.


## [Day 2: Dive!](https://adventofcode.com/2021/day/2): part one

You can click on the title above to read the full text of the puzzle. The TLDR version is:

You are given a dataset in the format

```
action amount
```

Where `action` is a string in the set `forward`, `down`, `up` and the amount is an integer to add/subtract to a dedicated counter. These counters represent the position on the horizontal plane and a depth (in the adventure, we are in a submarine). For example:

```
forward 5
down 5
forward 8
up 3
down 8
forward 2
```

`forward 5` increases by `5` the horizontal position, `down 5` **adds** 5 to the depth, and `up 3` **decreases** the depth by 3. The objective is to compute the final horizontal position and depth and multiply these values together. In the example, we end up with a horizontal position of 15 and a depth of 15, for a final result of **150**.


### Design phase

Being very very similar to the task solved for day one, the [same considerations](/tensorflow/2021/12/11/advent-of-code-tensorflow/#design-phase) about the sequential nature and the statefulness of the TensorFlow program hold.

This puzzle is not only sequential but also involves the creation of a mapping between the `action` and the counter to increment.

There are many different ways of implementing this mapping, TensorFlow provides us all the tooling needed. For example, we could use a [MutableHashTable](https://www.tensorflow.org/api_docs/python/tf/lookup/experimental/MutableHashTable) to map the actions to the counters, but being in the `experimental` module I'd suggest avoiding using it.
Instead, we can implement our own very, very coarse mapping method using a Python `Enum`.

The peculiarity of the Enum usage in TensorFlow is that we **cannot** use the basic `Enum` type, because it has no TensorFlow data-type equivalent. TensorFlow needs to know **everything** about the data its manipulating, and therefore it prevents the creation of Enums. We, have to use basic TensorFlow types like `tf.int64` and, thus, use an `Enum` specialization like [`IntEnum`](https://docs.python.org/3/library/enum.html#enum.IntEnum).


```python
from enum import IntEnum, auto

class Action(IntEnum):
    """Action enum, to map the direction read to an action to perform."""

    INCREASE_HORIZONTAL = auto()
    INCREASE_DEPTH = auto()
    DECREASE_DEPTH = auto()
```

### Input pipeline

Exactly like during [day 1](/tensorflow/2021/12/11/advent-of-code-tensorflow/#input-pipeline) we can use a `TextLineDataset` to read all the lines of the input dataset. This time, while we read the lines we need to apply a slightly more complicated mapping function. I defined the `_processor` function to `split` every line, map the action string to the corresponding `Action` enum value, and convert the amount to a `tf.int64`. The function returns the pair `(Action, amount)`.


```python
def _processor(line):
    splits = tf.strings.split(line)
    direction = splits[0]
    amount = splits[1]

    if tf.equal(direction, "forward"):
        action = Action.INCREASE_HORIZONTAL
    elif tf.equal(direction, "down"):
        action = Action.INCREASE_DEPTH
    elif tf.equal(direction, "up"):
        action = Action.DECREASE_DEPTH
    else:
        action = -1
    #    tf.debugging.Assert(False, f"Unhandled direction: {direction}")

    amount = tf.strings.to_number(amount, out_type=tf.int64)
    return action, amount

dataset = tf.data.TextLineDataset("input").map(_processor)
```

Note that every function passed to every method of a `tf.data.Dataset` instances **will always be executed in graph mode**. That's why I had to add the `action = -1` statement in the last `else` branch (and I couldn't use the assertion). In fact, the `if` statement is converted to a [`tf.cond`](https://www.tensorflow.org/api_docs/python/tf/cond?hl=en) by autograph, and since in the `true_fn` we modify a node (`action`) defined outside the `tf.cond` call, autograph forces us to modify the same node also in the else statement, in order to be able to create a direct acyclic graph in every scenario.

Moreover, I can use the `-1` without any problem because the `IntEnum` members are always positive when declared (as I did) with `auto()`, hence `-1` is an always invalid item (and a condition that will never be reached).

### Position counter

Similar to the solution of [day 1](/tensorflow/2021/12/11/advent-of-code-tensorflow/#counting-increments), we define our `PositionCounter` as a complete TensorFlow program with 2 states (variables). A variable for keeping track of the horizontal position and another for the depth.

```python

class PositionCounter(tf.Module):
    """Stateful counter. Get the final horizontal position and depth."""

    def __init__(self):
        self._horizontal_position = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._depth = tf.Variable(0, trainable=False, dtype=tf.int64)

    @tf.function
    def __call__(self, dataset: tf.data.Dataset) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            dataset: dataset yielding tuples (action, value), where action is
                     a valida Action enum.
        Returns:
            (horizontal_position, depth)
        """
        for action, amount in dataset:
            if tf.equal(action, Action.INCREASE_DEPTH):
                self._depth.assign_add(amount)
            elif tf.equal(action, Action.DECREASE_DEPTH):
                self._depth.assign_sub(amount)
            elif tf.equal(action, Action.INCREASE_HORIZONTAL):
                self._horizontal_position.assign_add(amount)
        return self._horizontal_position, self._depth
```

The `call` method accepts our `tf.data.Dataset` that yields tuples and performs the correct action on the states.

### Execution

```python
counter = PositionCounter()
horizontal_position, depth = counter(dataset)
result = horizontal_position * depth
tf.print("[part one] result: ", result)
```

Just create an instance of the `PositionCounter` and call it over the dataset previously created. Using type annotations while defining our functions simplifies their usage.

A **limitation** of the type annotation when used with TensorFlow is pretty easy to spot: we only have the `tf.Tensor` type, and the information of the TensorFlow data type (e.g. `tf.int64`, `tf.string`, `tf.bool`, ...) is not available.

Anyway, the execution gives the correct result :) and this brings us to part 2.

## [Day 2: Dive!](https://adventofcode.com/2021/day/2): part two

TLDR: there's a small modification of the challenge, that adds a new state. In short, there's now to keep track of another variable called `aim`. Now, `down`, `up`, and `forward` have a different meanings:

- down X increases your aim by X units.
- up X decreases your aim by X units.
- forward X does two things:
    - It increases your horizontal position by X units.
    - It increases your depth by your aim multiplied by X.

### Design phase - part two

It's just a matter of extending the `Action` enum defined at the beginning, adding the `_aim` variable, and acting accordingly to the new requirements in the `call` method. Without rewriting the complete solution, we can just expand the enum as follows

```python
class Action(IntEnum):
    """Action enum, to map the direction read to an action to perform."""

    INCREASE_HORIZONTAL = auto()
    INCREASE_DEPTH = auto()
    DECREASE_DEPTH = auto()
    INCREASE_AIM = auto()
    DECREASE_AIM = auto()
    INCREASE_HORIZONTAL_MUTIPLY_BY_AIM = auto()
```

And having defined the `_aim` variable in the same way of the `_depth` and `_horizontal_position`, update the `call` body as follows:

```python
for action, amount in dataset:
    if tf.equal(action, Action.INCREASE_DEPTH):
        self._depth.assign_add(amount)
    elif tf.equal(action, Action.DECREASE_DEPTH):
        self._depth.assign_sub(amount)
    elif tf.equal(action, Action.INCREASE_HORIZONTAL):
        self._horizontal_position.assign_add(amount)
    elif tf.equal(action, Action.INCREASE_HORIZONTAL_MUTIPLY_BY_AIM):
        self._horizontal_position.assign_add(amount)
        self._depth.assign_add(self._aim * amount)
    elif tf.equal(action, Action.DECREASE_AIM):
        self._aim.assign_sub(amount)
    elif tf.equal(action, Action.INCREASE_AIM):
        self._aim.assign_add(amount)
return self._horizontal_position, self._depth
```

### Execution - part two

The execution is identical to the previous one, there's no need to repeat it here. So, also the second puzzle is gone :)

## Conclusion

You can see the complete solution in folder `2` on the dedicated Github repository: [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

I created two different files for the two different parts, only because I'm lazy. There's for sure a more elegant solution without 2 different programs, but since the challenge was more or less a copy-paste of the one faced on day 1 I wasn't stimulated enough to write a single elegant solution.

The day 2 exercise is really easy and almost identical to the one presented on day 1. Also from the TensorFlow side, there are not many peculiarities to highlight.

I showed how to use Enums (in short, we can only use types that are TensorFlow compatible, hence no pure python Enums) and I presented a limitation that's related to the missing typing of the `tf.Tensor` in the type annotations, but that's all.

Luckily, I've already completed the challenges for days 3 and 4. Both of them are interesting, and in both, we'll see some interesting features of TensorFlow. A little spoiler, we'll use the `tf.TensorArray` :)

The next article about my pure TensorFlow solution for day 3 will arrive soon!

For any feedback or comment, please use the Disqus form below - thanks!
