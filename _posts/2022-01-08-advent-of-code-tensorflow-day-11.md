---
layout: post
title: "Advent of Code 2021 in pure TensorFlow - day 11"
date: 2022-01-08 08:00:00
categories: tensorflow
summary: "The Day 11 problem has lots in common with Day 9. In fact, will re-use some computer vision concepts like the pixel neighborhood, and we'll be able to solve both parts in pure TensorFlow by using only a tf.queue as a support data structure."
authors:
    - pgaleone
---

The Day 11 problem has lots in common with [Day 9](/tensorflow/2022/01/01/advent-of-code-tensorflow-day-9/). In fact, will re-use some computer vision concepts like the pixel neighborhood, and we'll be able to solve both parts in pure TensorFlow by using only a `tf.queue` as a support data structure.


## [Day 11: Dumbo Octopus](https://adventofcode.com/2021/day/11)

You can click on the title above to read the full text of the puzzle. The TLDR version is:

Our dataset is a grid of numbers. Every number represents an **energy level** of an octopus. On every time step, every energy level increases by 1 unit, and when the energy level goes beyond 9, the octopus **flashes**.
When an octopus flashes it increases the energy level of **all** (in every direction) the octopus in the neighborhood. This may trigger a cascade of flashes. Finally, any octopus that flashed during this time step, rests its energy level to 0.

For example:

<pre><code>Before any steps:
11111
19991
19191
19991
11111

After step 1:
34543
4<b>000</b>4
5<b>000</b>5
4<b>000</b>4
34543

After step 2:
45654
51115
61116
51115
45654
</code></pre>

An octopus is **highlighted** when it flashed during the given time step. Part one asks us to find out *how many total flashes are there after 100 steps?*

### Design phase: part one

The problem requires modeling the increment on every time step, and the propagation of the increment to the neighborhood when an energy level goes beyond 9. Thus, considering the input dataset as a 2D grayscale image, we can treat every energy level as a pixel. Thus, we need:

1. Define a function that given a pixel returns the 8-neighborhood coordinates. Differently from the solution presented in [day 9: design phase](/tensorflow/2022/01/01/advent-of-code-tensorflow-day-9/#design-phase-part-one), where the problem required us to only look at the 4-neighborhood, this time we need to consider also the diagonal pixels. Moreover, we **can't** simplify in the same way the problem **padding** the input image, since every update may trigger a **cascade** of updates that may involve the padding pixel introduced, and we don't want it.
2. Given that every flash may trigger a cascade of flash, we need to increment by 1 every value in the neighborhood of a flashing pixel and keep track of all the neighbors that exceeded the value of 9. Pushing the coordinates into a queue, and repeating the process for every pixel in the queue until the propagation is not complete.

### Input pipeline

We create a `tf.data.Dataset` object for reading the text file line-by-line [as usual](/tensorflow/2021/12/11/advent-of-code-tensorflow/#input-pipeline). Since we want to work with all the pixels at once, we convert the dataset in a 2D `tf.Tensor` - this is our octopus **population**.

```python
population = tf.convert_to_tensor(
    list(
        tf.data.TextLineDataset("fake")
        .map(tf.strings.bytes_split)
        .map(lambda string: tf.strings.to_number(string, out_type=tf.int64))
    )
)
```

We can now start implementing our TensorFlow program for solving part 1. We can start by the definition of the `_neighs` function.

### The 8-neighborhood

The first point of the [design phase](#design-phase-part-one) explains why we need to take care of the 8-neighborhood without padding the input. Hence, our neighborhood will contain `8` pixels if we are **inside** the image. It will contain `3` or `5` pixels of we are on a corner or on the side of the image, respectively.

We need to take care of these conditions manually.

```python
@staticmethod
@tf.function
def _neighs(grid: tf.Tensor, center: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    y, x = center[0], center[1]

    shape = tf.shape(grid, tf.int64) - 1

    if tf.logical_and(tf.less(y, 1), tf.less(x, 1)):  # 0,0
        mask = tf.constant([(1, 0), (0, 1), (1, 1)])
    elif tf.logical_and(tf.equal(y, shape[0]), tf.equal(x, shape[1])):  # h,w
        mask = tf.constant([(-1, 0), (0, -1), (-1, -1)])
    elif tf.logical_and(tf.less(y, 1), tf.equal(x, shape[1])):  # top right
        mask = tf.constant([(0, -1), (1, 0), (1, -1)])
    elif tf.logical_and(tf.less(x, 1), tf.equal(y, shape[0])):  # bottom left
        mask = tf.constant([(-1, 0), (-1, 1), (0, 1)])
    elif tf.less(x, 1):  # left
        mask = tf.constant([(1, 0), (-1, 0), (-1, 1), (0, 1), (1, 1)])
    elif tf.equal(x, shape[0]):  # right
        mask = tf.constant([(-1, 0), (1, 0), (0, -1), (-1, -1), (1, -1)])
    elif tf.less(y, 1):  # top
        mask = tf.constant([(0, -1), (0, 1), (1, 0), (1, -1), (1, 1)])
    elif tf.equal(y, shape[1]):  # bottom
        mask = tf.constant([(0, -1), (0, 1), (-1, 0), (-1, -1), (-1, 1)])
    else:  # generic
        mask = tf.constant(
            [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        )

    coords = center + tf.cast(mask, tf.int64)
    neighborhood = tf.gather_nd(grid, coords)
    return neighborhood, coords
```

The `_neihgs` function accepts the `grid` (2D `tf.Tensor`) and the coordinate of a pixel, that's the `center` of the neighborhood.

Depending on the `center` coordinate respect to the grid, the function builds a different mask used to create the pixel coordinate and gather the value in that coordinates.

Note that TensorFlow coordinates, by convention, are in `y,x` format and **not** in `x,y` format. We are now ready for implementing the algorithm.

### Simulating the population behavior

The puzzle asks us to simulate 100 steps, and count how many flashes happened during them. Thus, we need to completely simulate the octopus population behavior over time.

We already have our data organized in a grid, and from the second point of the [design phase](#design-phase-part-one), we know that the unique data structure needed for model the propagation is a `tf.queue`.

```python
def __init__(self, population, steps):
    super().__init__()

    self._steps = steps
    self._population = tf.Variable(population, dtype=tf.int64)
    self._counter = tf.Variable(0, dtype=tf.int64)

    self._zero = tf.constant(0, dtype=tf.int64)
    self._one = tf.constant(1, dtype=tf.int64)
    self._nine = tf.constant(9, tf.int64)
    self._ten = tf.constant(10, dtype=tf.int64)

    self._queue = tf.queue.FIFOQueue(-1, [tf.int64])

    self._flashmap = tf.Variable(tf.zeros_like(self._population))
```

The constructor requires the population and the number of steps to simulate. The `population` input variable is then used to initialize a `tf.Variable` that we'll use to store the population status after every time step.

The `_counter` variable will be our puzzle output, and the `_flashmap` is a helper variable with the same shape of the population that we use to keep track of the flashing pixels during every time step - so we can know what pixel flashed and keep track of them.

Other than the `_queue` used for modeling the flash propagation, we also define some constant with the dtype `tf.int64` that we'll later use inside the TensorFlow program. This is a good practice to follow since we avoid the creation of several different (and useless) constants in the graph, and we'll always refer to the same constants instead.

Modeling the population behavior is straightforward - it's just a matter of converting in TensorFlow code the puzzle instructions.

```python
@tf.function
def __call__(self):
    for step in tf.range(self._steps):
        # First, the energy level of each octopus increases by 1.
        self._population.assign_add(tf.ones_like(self._population))

        # Then, any octopus with an energy level greater than 9 flashes.
        flashing_coords = tf.where(tf.greater(self._population, self._nine))
        self._queue.enqueue_many(flashing_coords)

        # This increases the energy level of all adjacent octopuses by 1, including octopuses that are diagonally adjacent.
        # If this causes an octopus to have an energy level greater than 9, it also flashes.
        # This process continues as long as new octopuses keep having their energy level increased beyond 9.
        # (An octopus can only flash at most once per step.)
        while tf.greater(self._queue.size(), 0):
            p = self._queue.dequeue()
            if tf.greater(self._flashmap[p[0], p[1]], 0):
                continue
            self._flashmap.scatter_nd_update([p], [1])

            _, neighs_coords = self._neighs(self._population, p)
            updates = tf.repeat(
                self._one,
                tf.shape(neighs_coords, tf.int64)[0],
            )
            self._population.scatter_nd_add(neighs_coords, updates)
            flashing_coords = tf.where(tf.greater(self._population, self._nine))
            self._queue.enqueue_many(flashing_coords)

        # Finally, any octopus that flashed during this step has its energy level set to 0, as it used all of its energy to flash.
        indices = tf.where(tf.equal(self._flashmap, self._one))
        if tf.greater(tf.size(indices), 0):
            shape = tf.shape(indices, tf.int64)
            updates = tf.repeat(self._zero, shape[0])
            self._counter.assign_add(shape[0])
            self._population.scatter_nd_update(indices, updates)

        self._flashmap.assign(tf.zeros_like(self._flashmap))

        # tf.print(step, self._population, summarize=-1)
    return self._counter
```

The code is the precise implementation of the puzzle instructions, the comments are part of the [original puzzle text](https://adventofcode.com/2021/day/11) and after every comment, there's the equivalent TensorFlow implementation.

### Execution

Here we go!

```python
steps = tf.constant(100, tf.int64)
flash_counter = FlashCounter(population, steps)
tf.print("Part one: ", flash_counter())
```

Part one is solved! Let's see what part two is about.

## Design phase: part 2

The propagation of the flashes causes a nice phenomenon: the octopuses are synchronizing! For example

<pre><code>
After step 194:
6988888888
9988888888
8888888888
8888888888
8888888888
8888888888
8888888888
8888888888
8888888888
8888888888

After step 195:
<b>0000000000</b>
<b>0000000000</b>
<b>0000000000</b>
<b>0000000000</b>
<b>0000000000</b>
<b>0000000000</b>
<b>0000000000</b>
<b>0000000000</b>
<b>0000000000</b>
<b>0000000000</b>
</code></pre>

Part two asks us to determine the **first step** during which all the octopuses flash.

### Part two implementation

Our algorithm already models the population behavior, hence we already have implemented all we need to solve part two.

In fact, part two solution is **identical** to part 1 solution with the `for` loop replaced with a `while`. In part one, we have a fixed number of steps, instead this time we re-use the `counter` variable to keep track of how many steps we performed until the while condition holds.

```python
@tf.function
def find_sync_step(self):
    # use count as step
    self._counter.assign(0)
    while tf.logical_not(tf.reduce_all(tf.equal(self._population, self._zero))):
         self._counter.assign_add(1)
         # First, the energy level of each octopus increases by 1.
         # [ very same code of previous solution, omitted!]
         # Full version:
         # https://github.com/galeone/tf-aoc/blob/main/11/main.py#L66
```

Since we re-used the same internal status of the `FlashCounter` class, we have two options. Create a new object for invoking the `find_sync_step` or speeding up the computation knowing that our search starts from step `100` used to solve part 1.

```python
tf.print("Part two: ", steps + flash_counter.find_sync_step())
```

Here we go! Challenge 11 is solved in pure TensorFlow pretty easily.

## Conclusion

You can see the complete solution in folder `11` on the dedicated Github repository: [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

Solving this problem has, once again, demonstrated how TensorFlow can be used to solve any kind of programming problem, and how we can think of TensorFlow as a different programming language, especially if we are able to write pure TensorFlow programs (hence `tf.function` graph-converted functions).

If you missed the articles about the previous days' solutions, here's a handy list:

- [Day 1](/tensorflow/2021/12/11/advent-of-code-tensorflow/)
- [Day 2](/tensorflow/2021/12/12/advent-of-code-tensorflow-day-2/)
- [Day 3](/tensorflow/2021/12/14/advent-of-code-tensorflow-day-3/)
- [Day 4](/tensorflow/2021/12/17/advent-of-code-tensorflow-day-4/)
- [Day 5](/tensorflow/2021/12/22/advent-of-code-tensorflow-day-5/)
- [Day 6](/tensorflow/2021/12/25/advent-of-code-tensorflow-day-6/)
- [Day 7](/tensorflow/2021/12/28/advent-of-code-tensorflow-day-7/)
- [Day 8](/tensorflow/2021/12/28/advent-of-code-tensorflow-day-8/)
- [Day 9](/tensorflow/2022/01/01/advent-of-code-tensorflow-day-9/)
- [Day 10](/tensorflow/2022/01/04/advent-of-code-tensorflow-day-10/)

The next article will be about my solution to Day 12 problem. In that article I'll show how to work with graphs in the "traditional" way: no neural networks, only an adjacency matrix, and a search algorithm implemented using recursion. However, `tf.function` can't be used to write recursive functions :( hence I end up writing a pure TensorFlow **eager** solution.

For any feedback or comment, please use the Disqus form below - thanks!
