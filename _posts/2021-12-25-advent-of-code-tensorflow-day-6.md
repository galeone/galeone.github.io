---
layout: post
title: "Advent of Code 2021 in pure TensorFlow - day 6"
date: 2021-12-25 08:00:00
categories: tensorflow
summary: "The day 6 challenge has been the first one that obliged me to completely redesign for part 2 the solution I developed for part 1. For this reason, in this article,  we'll see two different approaches to the problem. The former will be computationally inefficient but will completely model the problem, hence it will be easy to understand. The latter, instead, will be completely different and it will focus on the puzzle goal instead of the complete modeling."
authors:
    - pgaleone
---

The day 6 challenge has been the first one that obliged me to completely redesign for part 2 the solution I developed for part 1. For this reason, in this article, we'll see two different approaches to the problem. The former will be computationally inefficient but will completely model the problem, hence it will be easy to understand. The latter, instead, will be completely different and it will focus on the puzzle goal instead of the complete modeling.

The second part will also use an experimental feature of TensorFlow I avoided during the [day 2 design phase](/tensorflow/2021/12/12/advent-of-code-tensorflow-day-2/#design-phase), but this time I had to since otherwise, the problem wouldn't be easily solvable. We'll also see how to correctly use `tf.TensorArray` in graph mode, and highlight an inconsistency of the TensorFlow framework.

## [Day 6: Lanternfish](https://adventofcode.com/2021/day/6)

You can click on the title above to read the full text of the puzzle. The TLDR version is:

You are asked to model the **exponential growth** of a population of lanternfish. Every lanternfish is modeled as a **timer** (number). Your puzzle input represents the population state at the initial state (time zero).

```
3,4,3,1,2
```

Every timer decreases its value by `1` day after day. The day after a timer reaches the `0` new lanternfish is generated and starts its counter from `8`. While the lanternfish that reached `0` resets its state to `6`. For example, after 11 days the population evolves in this fashion

```
Initial state: 3,4,3,1,2
After  1 day:  2,3,2,0,1
After  2 days: 1,2,1,6,0,8
After  3 days: 0,1,0,5,6,7,8
After  4 days: 6,0,6,4,5,6,7,8,8
After  5 days: 5,6,5,3,4,5,6,7,7,8
After  6 days: 4,5,4,2,3,4,5,6,6,7
After  7 days: 3,4,3,1,2,3,4,5,5,6
After  8 days: 2,3,2,0,1,2,3,4,4,5
After  9 days: 1,2,1,6,0,1,2,3,3,4,8
After 10 days: 0,1,0,5,6,0,1,2,2,3,7,8
After 11 days: 6,0,6,4,5,6,0,1,1,2,6,7,8,8,8
```

After `11` days there are a total of `16` fish. After 80 days there would be `5934` fish. The puzzle asks us to find how many fish will be present after 80 days given a different initial state (the puzzle input).

### Design phase: modeling the population

The problem clearly asks us to think about only the **number of fish** after a certain amount of days, but for part 1 we can model the population instead. Day by day, modeling the growth exactly ad presented in the example.

We can define an `evolve` function that takes the initial state as input, together with the number of days to model. This function uses a `tf.TensorArray` (dynamic shape data structure) to store the state after every iteration.

On every iteration, we only need to check for the `0` values in the array, replace them with the same amount of `6`, and append the very same amount of `8` at the end of the `TensorArray`.

The function is, thus, able to model the population growth and return the complete population state as output.

### Input pipeline

We create a `tf.data.Dataset` object for reading the text file line-by-line [as usual](/tensorflow/2021/12/11/advent-of-code-tensorflow/#input-pipeline). Moreover, since the dataset is just a single line (the initial state) we can convert it to an iterable (using `iter`) and extract the `initial_state` with `next`.

```python
initial_state = next(
    iter(
        tf.data.TextLineDataset("input")
        .map(lambda string: tf.strings.split(string, ","))
        .map(lambda numbers: tf.strings.to_number(numbers, out_type=tf.int64))
        .take(1)
    )
)
```

### Modeling the population

As introduced in the [design phase](#design-phase-modeling-the-population) we'll solve part 1 completely modeling the population.

The algorithm is precisely what has been described in the problem requirements.

```python
@tf.function
def evolve(initial_state: tf.Tensor, days: tf.Tensor):
    ta = tf.TensorArray(tf.int32, size=tf.size(initial_state), dynamic_size=True)
    ta = ta.unstack(initial_state)

    for _ in tf.range(1, days + 1):
        yesterday_state = ta.stack()
        index_map = tf.equal(yesterday_state, 0)
        if tf.reduce_any(index_map):
            indices = tf.where(index_map)
            transition_state = tf.tensor_scatter_nd_update(
                yesterday_state - 1,
                indices,
                tf.cast(tf.ones(tf.shape(indices)[0]) * 6, tf.int32),
            )
            ta = ta.unstack(transition_state)
            new_born = tf.reduce_sum(tf.cast(index_map, tf.int32))
            for n in tf.range(new_born):
                ta = ta.write(tf.size(transition_state, tf.int32) + n, 8)
        else:
            transition_state = yesterday_state - 1
            ta = ta.unstack(transition_state)
        today_state = ta.stack()
        # tf.print("after ", day, "days: ", today_state, summarize=-1)
    return today_state
```

I made extensive use of the `unstack` method of `tf.TensorArray` for completely overwriting the array content with the values. This is a heavy operation since it completely rewrites the content, and allocates new space for the new elements to write in new memory locations.

The algorithm is pretty easy: look at yesterday's state and check for zeros. If no zeros are present, just decrement all the timers and overwrite the array value.
If zeros are present, instead:

1. Keep track of their position (`indices`)
2. Replace them with `6` (create a `transition_status')
3. Update the `TensorArray`
4. Count how many zeros were present (`new_born`)
5. Append an `8` for every `new_born` to the `TensorArray`

After `days` iterations, the `tf.TensorArray` contains the complete model of the population status on the final day. With `stack`, we can convert the `tf.TensorArray` to a `tf.Tensor` and return it.

### TensorArray the correct usage in graph-mode

A careful reader may have noticed that we annotated with `@tf.function` the `evolve` function and it's working fine. This may sound strange since while solving the [day 3 puzzle](/tensorflow/2021/12/14/advent-of-code-tensorflow-day-3/) I highlighted the [TensorArray limitation in graph-mode](/tensorflow/2021/12/14/advent-of-code-tensorflow-day-3/#tensorarray-limitation-in-graph-mode).

Well, the limitation exists but it's really subtle. The difference between the usage of `tf.TensorArray` during Day 3 and above, it's the **location** of the declaration.

In part 3 I defined the TensorArray as an attribute, declaring it in the `__init__`:

```python
self._ta = tf.TensorArray(
        size=1, dtype=tf.int64, dynamic_size=True, clear_after_read=True
    )
```

and used it later inside a method. I did this on purpose because a `tf.TensorArray` is an object that creates a state (like `tf.Variable`) and as such, it should be declared outside the method using it.

This time, instead, I declared it inside the function itself and I re-create the object every time the function is called. It works fine!

This is a very particular behavior, since `tf.TensorArray` is a stateful object. I can update it, change its shape, re-use it whenever I need it. Moreover, [we know](/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/#handling-states-breaking-the-function-scope) that objects that create a state should be defined outside the function scope.

It looks like, instead, that `tf.TensorArray` needs to be declared and used within the **same scope**. I haven't found a valid explanation honestly, but I guess is some of the inconsistencies present in the framework. :\

### Execution

We modeled the state, but we are interested only in the total number of fish after 80 days. Hence the output is just the `size` of the resulting `tf.Tensor`

```python
days = tf.constant(80, tf.int64)
last_state = evolve(initial_state, days)
tf.print("# fish after ", days, " days: ", tf.size(last_state))
```

Day 6 puzzle 1 solved... but it's slow! Part 2 asks us an identical question:

> How many lanternfish would there be after 256 days?

If we run the very same code using ` tf.constant(256, tf.int64)` the process hangs for minutes until... it goes out of memory and the process gets killed by the OS.

We need to design a completely different solution.

## Design phase: part 2

Instead of modeling the population growth, perhaps it's better to focus **exactly** on what the text asks: **the number of fish**.

Perhaps it exists a [closed-form solution](https://en.wikipedia.org/wiki/Closed-form_expression) to this problem, or perhaps it exists a different way of observing the problem for understanding how to model it.

We are interested in the number of fish after a certain amount of days. Hence let's look at this value:

| Day | Status | Number of fish|
| --- | --- | --- |
| 0 |3,4,3,1,2 | 5 |
| 1 |2,3,2,0,1 | 5 |
| 2 |1,2,1,6,0,8 | 6 |
| 3 |0,1,0,5,6,7,8 | 7 |
| 4 |6,0,6,4,5,6,7,8,8 | 9 |

From this point of view, all we can see (and we already know) is that when a counter reaches `0`, on the next day the number of fish increases by the same number of zeros. Maybe we should look at **how many** fish are in a certain state?

| Day | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- |
| 0     | 0 | 1 | 1 | 2 | 1 | 0 | 0 | 0 | 0 |
| 1     | 1 | 1 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| 2     | 1 | 2 | 1 | 0 | 0 | 0 | 1 | 0 | 1 |
| 3     | 2 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 0 |
| 4     | 1 | 0 | 0 | 0 | 1 | 1 | 3 | 0 | 2 |

Can you see the progression in the rows `[0-5]` and `[7-8]`? I'll highlight some.

| Day | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- |
| 0     | 0 | 1 | 1 | **2** | 1 | 0 | 0 | 0 | 0 |
| 1     | 1 | 1 | **2** | 1 | 0 | 0 | 0 | 0 | 0 |
| 2     | 1 | **2** | 1 | 0 | 0 | 0 | **1** | 0 | **1** |
| 3     | **2** | 1 | 0 | 0 | 0 | **1** | 1 | **1** | 0 |
| 4     | 1 | 0 | 0 | 0 | **1** | 1 | 3 | 0 | 2 |

Looking at the problem from this perspective it's way more clear how to model the number of fish! In fact, if we keep track of the number of fish in a certain state, for each day, we'll see that the number of fish that yesterday were in status `X` today is in status `X-1`, tomorrow will be in status `X-2`, and so on until they reach `X=0`.

When this happens (on the next day), the number of fish in status `6` (previously in status `7`) is increased by the amount of fish that were in status `0`, and at the same time the same number of fish spawn with status `8`.

In practice, we just need a table, shift the values to the left, and when there are zeroes execute the iteration step just described.

## Modeling the number of fish

TensorFlow, luckily, offers to correct data structure to model this table: a mutable hash table (`tf.lookup.experimental.MutableHashTable`). Even if experimental, this is the only (easy) way we have to represent the table modeled in the design phase.

Since the number of fish grows exponentially, we must use `tf.int64` as the default data type. **Exactly like `tf.TensorArray`** the `MutableHashTable` must be declared and used in the `@tf.function`-decorated method, and it **can't** be declared in the init and used in the method.

```python
class TableCounter(tf.Module):
    def __init__(self):
        super().__init__()

        self._zero = tf.constant(0, tf.int64)
        self._one = tf.constant(1, tf.int64)
        self._six = tf.constant(6, tf.int64)
        self._eight = tf.constant(8, tf.int64)
        self._nine = tf.constant(9, tf.int64)

    @tf.function
    def count(self, initial_state: tf.Tensor, days: tf.Tensor):
        # BUG. There's ne key int32 with value int64 :<
        # Must use both int64
        # NOTE NOTE NOTE!!
        # Like TensorArrays, the hashmap gives the error:
        # Cannot infer argument `num` from shape <unknown>
        # If declared in the init (self._hasmap) and then used
        # The definition should be here mandatory for this to work.

        hashmap = tf.lookup.experimental.MutableHashTable(
            tf.int64, tf.int64, self._zero
        )

        keys, _, count = tf.unique_with_counts(initial_state, tf.int64)
        hashmap.insert(keys, count)
```

Being experimental, `MutableHashTable` is buggy. In fact, it would have made sense to use an `int32` (or even `int8`) as the key data type, but it's not possible (yet).

The `hashmap` is initialized (at day 0) with the `initial_state` count for each lanternfish in a certain state: for doing it, the `tf.unique_with_counts` function is very helpful.

Now, we only need to iterate for the requested number of days and implement the algorithm previously described.

```python
for _ in tf.range(self._one, days + self._one):
    # NOTE: This has no defined shape if the map is not defined in this method!!
    yesterday_state = hashmap.lookup(tf.range(self._nine))
    if tf.greater(yesterday_state[0], self._zero):
        # handled values in keys [0, 5], [7, 8]
        today_state = tf.tensor_scatter_nd_update(
            yesterday_state,
            tf.concat(
                [
                    tf.reshape(tf.range(self._eight), (8, 1)),
                    [[self._eight]],
                ],
                axis=0,
            ),
            tf.concat(
                [
                    hashmap.lookup(tf.range(self._one, self._nine)),
                    [yesterday_state[0]],
                ],
                axis=0,
            ),
        )
        # Add the number of zeros as additional number of six
        today_state = tf.tensor_scatter_nd_add(
            today_state, [[self._six]], [yesterday_state[0]]
        )
    else:
        # shift the the left all the map
        # put a 0 in 8 position

        updates = tf.concat(
            [
                tf.unstack(
                    tf.gather(yesterday_state, tf.range(self._one, self._nine))
                ),
                [self._zero],
            ],
            axis=0,
        )
        indices = tf.reshape(tf.range(self._nine), (9, 1))
        today_state = tf.tensor_scatter_nd_update(
            yesterday_state, indices, updates
        )

    hashmap.insert(tf.range(self._nine), today_state)
return tf.reduce_sum(hashmap.lookup(tf.range(self._nine)))
```

The code above implements the previously described algorithm. Using the `TableCounter` object is straightforward:

```python
days = tf.constant(256, tf.int64)
counter = TableCounter()
tf.print("# fish after ", days, " days: ", counter.count(initial_state, days))
```

Problem 6 is now **efficiently** solved!

## Conclusion

You can see the complete solution in folder `6` on the dedicated Github repository: [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

Solving part 2 of the problem required a complete redesign of the part 1 solution, completely changing the perspective over the problem. Implementing both parts required the usage of not-widely-used TensorFlow features like the `MutableHashTable` and the `TensorArray`. These data structures, even if conceptually able to have a state **must** be declared and used in the `@tf.function`-decorated method and **can't** be declared in the `__init__`, otherwise this error message happens during the graph-creation

> Cannot infer argument `num` from shape <unknown>.

I'm still trying to understand if this behavior has some kind of justification or if it's a bug.

The challenge in the challenge of using only TensorFlow for solving the problem is slowly progressing, so far I solved all the puzzles up to Day 10 (inclusive). So get ready for at least 4 more articles :) Let's see when (and if!) TensorFlow alone won't be enough.

If you missed the articles about the previous days' solutions, here's a handy list:

- [Day 1](/tensorflow/2021/12/11/advent-of-code-tensorflow/)
- [Day 2](/tensorflow/2021/12/12/advent-of-code-tensorflow-day-2/)
- [Day 3](/tensorflow/2021/12/14/advent-of-code-tensorflow-day-3/)
- [Day 4](/tensorflow/2021/12/17/advent-of-code-tensorflow-day-4/)
- [Day 5](/tensorflow/2021/12/22/advent-of-code-tensorflow-day-5/)

The next article will be about my solution to [Day 7](https://adventofcode.com/2021/day/7) problem. For solving the second part, I have used a very nice feature of TensorFlow the **ragged tensors** - that are **not** sparse tensors, but a superset of the standard tensor representation. Stay tuned for the next one!

For any feedback or comment, please use the Disqus form below - thanks!
