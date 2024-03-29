---
layout: post
title: "Advent of Code 2021 in pure TensorFlow - day 5"
date: 2021-12-22 08:00:00
categories: tensorflow
summary: "The day 5 challenge is easily solvable in pure TensorFlow thanks to its support for various distance functions and the power of the tf.math package. The problem only requires some basic math knowledge to be completely solved - and a little bit of computer vision experience doesn't hurt."
authors:
    - pgaleone
---

The day 5 challenge is easily solvable in pure TensorFlow thanks to its support for various distance functions and the power of the `tf.math` package. The problem only requires some basic math knowledge to be completely solved - and a little bit of computer vision experience doesn't hurt.

## [Day 5: Hydrothermal Venture](https://adventofcode.com/2021/day/5)

You can click on the title above to read the full text of the puzzle. The TLDR version is:

The puzzle input contains segments coordinates, in the format `x1,y1 -> x2,y2`, like

```
0,9 -> 5,9
8,0 -> 0,8
9,4 -> 3,4
2,2 -> 2,1
7,0 -> 7,4
```

A segment is not only start and end points, but it is made of all the points that connect start and end. 

- An entry like `1,1 -> 1,3` covers points `1,1`, `1,2`, and `1,3`.
- An entry like `9,7 -> 7,7` covers points `9,7`, `8,7`, and `7,7`.

The puzzle asks us to "draw" all the lines and **count the number of points where at least two lines overlap**.

Part 1 asks us to focus only on the horizontal and vertical lines (`x1 = x2` or `y1 = y2`), while (spoiler) part 2 asks to consider also diagonal segments at exactly `45` degrees.

We'll design a solution for both parts, that will compute the correct output depending on a boolean flag.

### Design phase

The problem is pretty trivial if one is used to working with pixels. In fact, the problem can be seen as the implementation of a segment drawing function over 2D gray-scale image.

- The 2D image resolution is given by the [minimum bounding box](https://en.wikipedia.org/wiki/Minimum_bounding_box) containing all the points.
- The segment orientation can be easily found by using the [atan2](https://en.wikipedia.org/wiki/Atan2) on the direction vector.
- The **pixel coordinates** to draw are the result of the [linear interpolation](https://en.wikipedia.org/wiki/Linear_interpolation) between the start and end points.
- The **pixel coordinates must be integers** since it makes no sense to draw in positions like (0.8, 1.5). This constraint is automatically solved by the problem constraints: in 2D grid segments at 45 degrees, horizontal, and vertical all of them have integer coordinates.
- The **number of pixels** produced by the interpolation should be precisely the [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance) (also called $$ L_\infty $$ distance or **chessboard distance**) between the start and end points.

The last point is the most important one since it constrains the number of pixel coordinates generated by the linear interpolation and makes the coordinates always integers.

### Input pipeline

We create a `tf.data.Dataset` object for reading the text file line-by-line [as usual](/tensorflow/2021/12/11/advent-of-code-tensorflow/#input-pipeline). All we need to do is to choose a handy representation for the elements produced at every iteration. I choose the traditional representation of a line using pairs of coordinates.

```python
def _get_segment(
    line: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    points = tf.strings.split(line, " -> ")
    p1 = tf.strings.split(points[0], ",")
    p2 = tf.strings.split(points[1], ",")

    x1 = tf.strings.to_number(p1[0], tf.int64)
    y1 = tf.strings.to_number(p1[1], tf.int64)
    x2 = tf.strings.to_number(p2[0], tf.int64)
    y2 = tf.strings.to_number(p2[1], tf.int64)
    return tf.convert_to_tensor((x1, y1)), tf.convert_to_tensor((x2, y2))

dataset = tf.data.TextLineDataset("input").map(_get_segment)
```

The `_get_segment` function (that it's executed in graph mode even if not decorated explicitly with `@tf.function`) strips the `->` and creates the pair of points: `((x1, y1), (x2, y2))`.

### Creating the Grid

A 2D image is a grid of pixels. The resolution of the image (the number of columns and rows of the grid) is given by the [minimum bounding box](https://en.wikipedia.org/wiki/Minimum_bounding_box) containing all the points.

Since we need to draw on this grid, this object should be a `tf.Variable`. We can thus define the `__init__` of our TensorFlow program in this way.

```python
class Grid(tf.Module):
    def __init__(self, dataset):
        super().__init__()
        bbox_w = tf.reduce_max(list(dataset.map(lambda p1, p2: (p1[0], p2[0])))) + 1
        bbox_h = tf.reduce_max(list(dataset.map(lambda p1, p2: (p1[1], p2[1])))) + 1
        self._grid = tf.Variable(
            tf.zeros((bbox_w, bbox_h), dtype=tf.int64), trainable=False
        )
        self._dataset = dataset
```

In short, we just searched the maximum coordinates along the x and y axes and added 1 because the given coordinates include zero as a valid location.

Before doing the actual drawing, we can implement an interpolation function that uses the Chebyshev norm.

### Chessboard interpolation

As presented in the [design](#design-phase) section, the 2D grid constraints the pixel coordinates to be integers. Using the Chebyshev distance is the correct way of finding the number of pixels to generate from the interpolation process.

TensorFlow allows us to interpolate along the x and y axes in parallel with a single function call.

```python
@staticmethod
@tf.function
def interpolate(p1: tf.Tensor, p2: tf.Tensor):
    """Linear interpolation from p1 to p2 in the discrete 2D grid.
    Args:
        p1: Tensor with values (x, y)
        p2: Tensor with values (x, y)
    Returns:
        The linear interpolation in the discrete 2D grid.
    """
    # +1 handles the case of p1 - p2 == 1
    norm = tf.norm(tf.cast(p1 - p2, tf.float32), ord=tf.experimental.numpy.inf) + 1
    return tf.cast(
        tf.math.ceil(tf.linspace(p1, p2, tf.cast(norm, tf.int64))), tf.int64
    )
```

Since the challenge in the challenge is to only use TensorFlow, I used `tf.experimental.numpy.inf` instead of `numpy.inf` for specifying the `inf` value required by the `tf.norm` function to use the Chessboard distance ( $$ L_\infty $$).

There's now only to implement the drawing logic.

### Drawing lines on the grid

Having decided to solve with the same code both parts of the puzzle, I can just define the `call` method accepting a boolean flag that will change the behavior of the method.

Being stateful, we need to remember to reset to 0 the grid state when the method is called, before starting looping over the `self._dataset`.

```python
@tf.function
def __call__(self, part_one: tf.Tensor) -> tf.Tensor:
    """Given the required puzzle part, changes the line drawing on the grid
    and the intersection couunt.
    Args:
        part_one: boolean tensor. When true, only consider straight lines and
                  a threshold of 1. When false, consider straight lines and diagonal
                  lines.
    Returns
        the number of intersections
    """
    self._grid.assign(tf.zeros_like(self._grid))
```

Now, we can change the loop behavior depending on the `part_one` value. We can define the logic as follows

- Use `tf.math.atan2` over the direction vector to get the angle in radians. Convert it to degrees and make it always positive by summing 360 if it's negative.
- If `part_one` is `True`: consider only horizontal and vertical lines.
- If `part_one` is `False`: check if the angle is exactly at 45 degrees or the lines are horizontal and vertical.

In both cases, if the condition holds, we can interpolate between the start and end points to get the `pixels` and draw them on the image using the `assign` method of `tf.Variable` and the `tf.tensor_scatter_nd_add` function for incrementing by 1 only the values in the `pixels` coordinate.

```python
    for start, end in self._dataset:
        # Discrete interpolation between start and end
        # part 1 requires to consider only straight lines
        # (x1 = x2 or y1 = y2)
        # but I guess (hope) doing the generic discrete interpolation
        # will simplify part 2 (no idea, just a guess)
        float_start = tf.cast(start, tf.float32)
        float_end = tf.cast(end, tf.float32)
        direction = float_start - float_end
        angle = (
            tf.math.atan2(direction[1], direction[0])
            * 180
            / tf.experimental.numpy.pi
        )
        if tf.less(angle, 0):
            angle = 360 + angle
        if tf.logical_or(
            tf.logical_and(
                tf.logical_not(part_one),
                tf.logical_and(
                    tf.logical_not(tf.equal(tf.math.mod(angle, 90), 0)),
                    tf.equal(tf.math.mod(angle, 45), 0),
                ),
            ),
            tf.logical_or(
                tf.equal(start[0], end[0]),
                tf.equal(start[1], end[1]),
            ),
        ):
            pixels = self.interpolate(start, end)
            self._grid.assign(
                tf.tensor_scatter_nd_add(
                    self._grid, pixels, tf.ones(tf.shape(pixels)[0], dtype=tf.int64)
                )
            )
```

At the end of the loop, we can just check how many values are greater than `1` and count them, to get the expected result.

```python
    # tf.print(tf.transpose(grid, perm=(1, 0)), summarize=-1)
    threshold = tf.constant(1, tf.int64)
    mask = tf.greater(self._grid, threshold)
    return tf.reduce_sum(tf.cast(mask, tf.int64))
```

### Execution

Here we go!

```python
grid = Grid(dataset)

tf.print("# overlaps (part one): ", grid(tf.constant(True)))
tf.print("# overlaps (part two): ", grid(tf.constant(False)))
```

Day 5 puzzle solved in both parts :)

## Conclusion

You can see the complete solution in folder `5` on the dedicated Github repository: [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

Solving this problem having experience in the computer vision domain has been easy since there are lots of concepts from that domain that can be applied in this problem resolution.

The challenge in the challenge of using only TensorFlow for solving the problem is slowly progressing, so far I solved all the puzzles up to Day 8 (inclusive). So get ready for at least 3 more articles :) Let's see when (and if!) TensorFlow alone won't be enough.

If you missed the articles about the previous days' solutions, here's a handy list:

- [Day 1](/tensorflow/2021/12/11/advent-of-code-tensorflow/)
- [Day 2](/tensorflow/2021/12/12/advent-of-code-tensorflow-day-2/)
- [Day 3](/tensorflow/2021/12/14/advent-of-code-tensorflow-day-3/)
- [Day 4](/tensorflow/2021/12/17/advent-of-code-tensorflow-day-4/)

The next article will be about my solution of [Day 6](https://adventofcode.com/2021/day/6) problem. I'll present 2 different solutions, one computationally intensive but easy to understand, and another developed for solving the part 2 - that's the very same problem of part 1 with a different input parameter. This small change will show how the previously developed solution is inefficient and requires us to approach the problem in a different way. Stay tuned!

For any feedback or comment, please use the Disqus form below - thanks!
