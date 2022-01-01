---
layout: post
title: "Advent of Code 2021 in pure TensorFlow - day 9"
date: 2022-01-01 08:00:00
categories: tensorflow
summary: "The day 9 challenge can be seen as a computer vision problem. TensorFlow contains some computer vision utilities that we'll use - like the image gradient - but it's not a complete framework for computer vision (like OpenCV). Anyway, the framework offers primitive data types like tf.TensorArray and tf.queue that we can use for implementing a flood-fill algorithm in pure TensorFlow and solve the problem."
authors:
    - pgaleone
---

The day 9 challenge can be seen as a computer vision problem. TensorFlow contains some computer vision utilities that we'll use - like the image gradient - but it's not a complete framework for computer vision (like OpenCV). Anyway, the framework offers primitive data types like tf.TensorArray and tf.queue that we can use for implementing a flood-fill algorithm in pure TensorFlow and solve the problem.

## [Day 9: Smoke Basin](https://adventofcode.com/2021/day/9)

You can click on the title above to read the full text of the puzzle. The TLDR version is:

Our dataset is a heightmap where each number corresponds to the height of a particular location. 9 is the maximum value, 0 is the lowest.

<pre>
2<b>1</b>9994321<b>0</b>
3987894921
98<b>5</b>6789892
8767896789
989996<b>5</b>678
</pre>

Our first goal is to find the low points - the locations that are lower than any of its adjacent locations. Most locations have four adjacent locations (up, down, left, and right); locations on the edge or corner of the map have three or two adjacent locations, respectively. (Diagonal locations do not count as adjacent.).

Once the low points have been found, we must compute the sum of the **risk levels**. The risk level for a low point is defined as the sum of 1 plus its height.

In the example above, we have 4 low points (highlighted) and the sum of their risk levels is 15.

### Design phase: part one

The heightmap can be seen as a 2D image. Every height is a pixel, and every pixel has is 4-neighborhood. We consider only the 4-neighborhood because we are asked to only look at horizontal and vertical neighbors and not to the diagonal ones.

We can loop over the image using a sliding window approach, centering a `3x3` kernel on every location and for each location searching for the lowest point.

Finding the low points requires looking at every pixel, extracting its 4-neighbors, and finding the minimum value among the 5 pixels (the neighbors and the pixel itself). For dealing with border pixels, we can just pad the input image with the correct amount of `9`s (hence, point with maximum value).


### Input pipeline

We create a `tf.data.Dataset` object for reading the text file line-by-line [as usual](/tensorflow/2021/12/11/advent-of-code-tensorflow/#input-pipeline).

```python
dataset = (
    tf.data.TextLineDataset("input")
    .map(tf.strings.bytes_split)
    .map(lambda string: tf.strings.to_number(string, out_type=tf.int32))
)
```

Since we consider the whole dataset as a single image, we can convert it as a 2D tensor. We'll do it directly in the constructor of the `Finder` object we are going to create for solving the problem. Moreover, since we are interested in the padded input image, we can also compute the padding amount and pad the image using `tf.pad`.

```python
class Finder(tf.Module):
    def __init__(self, dataset: tf.data.Dataset):

        super().__init__()

        self._dataset = dataset
        self._image = tf.convert_to_tensor(list(self._dataset))
        self._shape = tf.shape(self._image)
        self._max_value = tf.reduce_max(self._image)

        if tf.not_equal(tf.math.mod(self._shape[0], 3), 0):
            pad_h = self._shape[0] - (self._shape[0] // 3 + 3)
        else:
            pad_h = 0
        if tf.not_equal(tf.math.mod(self._shape[1], 3), 0):
            pad_w = self._shape[1] - (self._shape[1] // 3 + 3)
        else:
            pad_w = 0

        self._padded_image = tf.pad(
            self._image,
            [[0, pad_w], [1, pad_h]],
            mode="CONSTANT",
            constant_values=self._max_value,
        )

        self._padded_shape = tf.shape(self._padded_image)
```

### Finding the 4-neighbors

We'll loop over the image in a sliding window fashion, and for every pixel, we'll search for the 4-neighborhood. Hence, we can define a function that given an input `grid` (2D Tensor), and the coordinates of pixel (in TensorFlow format, hence `(y,x)`) it returns both the neighborhood values and the neighborhood coordinates.

```python
@tf.function
def _four_neigh(
    self, grid: tf.Tensor, center: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    neigh_mask = tf.constant([(-1, 0), (0, -1), (1, 0), (0, 1)])
    y, x = center[0], center[1]

    if tf.logical_and(tf.less(y, 1), tf.less(x, 1)):
        mask = neigh_mask[2:]
    elif tf.less(y, 1):
        mask = neigh_mask[1:]
    elif tf.less(x, 1):
        mask = tf.concat([[neigh_mask[0]], neigh_mask[2:]], axis=0)
    else:
        mask = neigh_mask

    coords = center + mask

    neighborhood = tf.gather_nd(grid, coords)
    return neighborhood, coords
```

The function just avoids looking in non-existent coordinates and gathers all the required values with `tf.gather_nd`.

We are now ready to loop over the padded image, center our "virtual" kernel (because we are not really defining a 3x3 kernel) on every pixel, find the neighborhood, and search for the lowest points.

### Finding the lowest points

Since we are creating a pure TensorFlow program, the variables must be defined outside the methods decorated with `tf.function`. Thus, since we are interested in the sum of the risk level, we'll add in the `init` a variable for this sum (`count`). We'll use a `tf.TensorArray` to save the coordinate of every lowest point found and we return them as well.

```python
def __init__(self, dataset: tf.data.Dataset):
    # [...]
    self._count = tf.Variable(0)

@tf.function
def low_points(self) -> Tuple[tf.Tensor, tf.Tensor]:
    self._count.assign(0)
    ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

    for y in tf.range(self._padded_shape[0] - 1):
        for x in tf.range(self._padded_shape[1] - 1):
            center = tf.convert_to_tensor([y, x])
            neighborhood, _ = self._four_neigh(self._padded_image, center)
            extended_neighborhood = tf.concat(
                [tf.expand_dims(self._padded_image[y, x], axis=0), neighborhood],
                axis=0,
            )

            minval = tf.reduce_min(extended_neighborhood)
            if tf.logical_and(
                tf.reduce_any(tf.not_equal(extended_neighborhood, minval)),
                tf.equal(minval, self._padded_image[y, x]),
            ):
                self._count.assign_add(1 + self._padded_image[y, x])

                ta = ta.write(ta.size(), center)

    return ta.stack(), self._count
```

### Execution

Here we go!

```python
finder = Finder(dataset)
tf.print("Part one: ", finder.low_points()[1])
```

Part one is easily solved. Let's see what part two is about.

## Design phase: part 2

Part 2 asks us to find the 3 largest **basins** and multiply their size together. A basin is all locations that eventually flow downward to a single low point. Therefore, every low point has a basin. The size of a basin is the number of locations within the basin.

In the previous example there are 4 basins. Here's the middle basin, of size 14:

<pre>
<code>2199943210
39<b>878</b>94921
9<b>85678</b>9892
<b>87678</b>96789
9<b>8</b>99965678
</code>
</pre>

and here's also the top-right basin of size 9:
<pre><code>21999<b>43210</b>
398789<b>4</b>9<b>21</b>
985678989<b>2</b>
8767896789
9899965678
</code></pre>

How can we find the basins? For sure we know that every low point is inside a basin, we can consider every low point a **seed** for a [flood fill algorithm](https://en.wikipedia.org/wiki/Flood_fill). But how can we find the basins thresholds? All we know is that every location with `9` is a natural threshold and that a basin is a flow of **decreasing numbers around a low point**.

Every computer vision practitioner immediately thinks to the [image gradient](https://en.wikipedia.org/wiki/Image_gradient#:~:text=An%20image%20gradient%20is%20a%20directional%20change%20in%20the%20intensity%20or%20color%20in%20an%20image.) when talking about **changes in the color intensity**. In particular, we can compute the horizontal and vertical gradients and compute the **gradient magnitude** for extracting the information about the rate of change in every pixel location.

For example, the gradient magnitude of the example image is:

<pre>
<code>
7 0 16 1 2 6 4 0 6 0
6 12 2 4 0 0 6 10 8 6
0 2 4 2 2 2 2 4 0 8 1
1 0 0 4 3 2 6 0 0 0 1
0 1 2 0 0 3 2 5 4 3 2
</code>
</pre>

The higher the magnitude the higher the change along the x and y directions. The gradient magnitude alone doesn't give us a clear indication of the basin thresholds, but it shows are where there are no changes (zeros) and where there are changes (ascending/descending values). If we combine this information, with the location of the natural barriers (where the 9 are), we end up with this image:

<pre>
<code>
<b>0  16</b> -1 -1 -1  <b>4  0  6  0 10</b>
<b>12</b> -1  <b>4  0  0</b> -1 <b>10</b> -1  <b>6  9</b>
-1  <b>4  2  2  2  2 </b>-1  <b>0</b> -1 14
 <b>0  0  4  3  2 </b>-1 <b> 0  0  0</b> -1
-1  <b>2</b> -1 -1 -1  <b>2  5  4  3  2</b>
</code>
</pre>

Where every 9 location has been replaced with a `-1`. The basins are perfectly detected :)

Now that we are able to create the artificial thresholds for the flood fill algorithm, we only need to implement it. For doing it we'll rely upon the `tf.queue.FIFOQueue` object.

### TensorFlow queues

The TensorFlow [`tf.queue`](https://www.tensorflow.org/api_docs/python/tf/queue) module offers several different implementations of the `tf.queue.QueueBase` object. The simples implementation to use is the [`FIFOQueue`](https://www.tensorflow.org/api_docs/python/tf/queue/FIFOQueue), that a queue implementation that dequeues elements in first-in first-out order.

Differently from the `tf.TensorArray`, we can treat the queue like a `tf.Variable` and declare it in the `init` and use in a `tf.function`-decorated method without any problem.

### Detecting the basins

From the [design phase: part two](#design-phase-part-2) section we know that we need to compute the image gradient and get the gradient magnitude. This is straightforward since TensorFlow has a ready-to-use method for doing it. Moreover, we know we need to implement a flood fill algorithm, hence we need a queue.

Thus, in the `init` we add the `_norm` variable that will contain the image gradient magnitude (initialized to -1), we also add the `queue` for the flood fill algorithm.

```python
def __init__(self, dataset: tf.data.Dataset):
    # [...]
    self._norm = tf.Variable(tf.zeros(self._padded_shape, dtype=tf.int32) - 1)
    self._queue = tf.queue.FIFOQueue(-1, [tf.int32])
```

We can now write the `basins` function, which finds all the basins, computes their size, and returns the product of the three largest basins as output.

```python
@tf.function
def basins(self) -> tf.Tensor:
    batch = tf.reshape(
        self._padded_image, (1, self._padded_shape[0], self._padded_shape[1], 1)
    )
    gradients = tf.squeeze(tf.image.image_gradients(batch), axis=1)

    y_grad, x_grad = gradients[0], gradients[1]

    # Gradient magnitude is constant where there are no changes
    # Increases or stray constants from the low point (seed)
    norm = tf.cast(tf.norm(tf.cast(y_grad + x_grad, tf.float32), axis=-1), tf.int32)
    # Set the basin thresholds to -1 (where the 9s are)
    norm = tf.where(tf.equal(self._padded_image, 9), -1, norm)
    self._norm.assign(norm)
```

The `tf.image.image_gradients` function works on a batch of images, hence we first need to reshape the image from a `tf.Tensor` with shape `(H,W)` to a tensor with shape `(1, H, W, 1)`.

With these few lines of code, we are in the situation shown in the previous design phase, with an image containing `-1` where the thresholds are, and the gradient magnitudes inside the basins.

Now, we can use the `low_points` method for getting the seeds for our flood fill algorithm and propagate them within the basin. We use a TensorArray to keep track of the three largest basins size and compute their product at the end.

```python
# For every se_posd, "propagate" in a flood fill-fashion.
# The -1s are the thresholds
seeds = self.low_points()[0]
ta = tf.TensorArray(tf.int32, size=3)
ta.unstack([0, 0, 0])
for idx in tf.range(2, tf.shape(seeds)[0] + 2):
    # Fill with idx (watershed like: different colors)
    seed = seeds[idx - 2]
    y = seed[0]
    x = seed[1]

    # Set the seed position to the label
    self._norm.scatter_nd_update([[y, x]], [-idx])

    # Find the 4 neighborhood, and get the values != -1
    neighborhood, neigh_coords = self._four_neigh(self._norm, seed)
    update_coords = tf.gather_nd(
        neigh_coords, tf.where(tf.not_equal(neighborhood, -1))
    )
    if tf.greater(tf.size(update_coords), 0):
        self._queue.enqueue_many(update_coords)
        while tf.greater(self._queue.size(), 0):
            pixel = self._queue.dequeue()
            # Update this pixel to the label value
            py, px = pixel[0], pixel[1]
            self._norm.scatter_nd_update([[py, px]], [-idx])
            px_neigh_vals, px_neigh_coords = self._four_neigh(self._norm, pixel)
            px_update_coords = tf.gather_nd(
                px_neigh_coords,
                tf.where(
                    tf.logical_and(
                        tf.not_equal(px_neigh_vals, -1),
                        tf.not_equal(px_neigh_vals, -idx),
                    )
                ),
            )
            if tf.greater(tf.size(px_update_coords), 0):
                self._queue.enqueue_many(px_update_coords)

    basin_size = tf.reduce_sum(tf.cast(tf.equal(self._norm, -idx), 3))
    if tf.greater(basin_size, ta.read(0)):
        first = basin_size
        second = ta.read(0)
        third = ta.read(1)
        ta = ta.unstack([first, second, third])
    elif tf.greater(basin_size, ta.read(1)):
        first = ta.read(0)
        second = basin_size
        third = ta.read(1)
        ta = ta.unstack([first, second, third])
    elif tf.greater(basin_size, ta.read(2)):
        ta = ta.write(2, basin_size)

return tf.reduce_prod(ta.stack())
```

If we print the content of `self._norm` after the execution of the algorithm, we can visualize the seed propagation.

<pre>
<code>
<b>-2</b> <b>-2</b> -1 -1 -1 <b>-3</b> <b>-3</b> <b>-3</b> <b>-3</b> <b>-3</b>
<b>-2</b> -1 <b>-4</b> <b>-4</b> <b>-4</b> -1 <b>-3</b> -1 <b>-3</b> <b>-3</b>
-1 <b>-4</b> <b>-4</b> <b>-4</b> <b>-4</b> <b>-4</b> -1 <b>-5</b> -1 <b>-3</b>
<b>-4</b> <b>-4</b> <b>-4</b> <b>-4</b> <b>-4</b> -1 <b>-5</b> <b>-5</b> <b>-5</b> -1
-1 <b>-4</b> -1 -1 -1 <b>-5</b> <b>-5</b> <b>-5</b> <b>-5</b> <b>-5</b>
</code>
</pre>

Here we go! Day 9 problem completely solved treating it as a computer vision problem :)

## Conclusion

You can see the complete solution in folder `9` on the dedicated Github repository: [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

Solving this problem has been really fun because reusing knowledge that comes from the computer vision domain (one of the domains I'm more involved in) is always exciting. In particular, treating the input as an image and, thus, solving the problem using the image gradients, the neighborhoods, the flood fill algorithms, and all the other computer vision-related concepts is really fascinating to me.

The challenge in the challenge of using only TensorFlow for solving the problem is slowly progressing, so far I solved all the puzzles up to Day 12 (inclusive). So get ready for at least 3 more articles :) Let's see when (and if!) TensorFlow alone won't be enough.

If you missed the articles about the previous days' solutions, here's a handy list:

- [Day 1](/tensorflow/2021/12/11/advent-of-code-tensorflow/)
- [Day 2](/tensorflow/2021/12/12/advent-of-code-tensorflow-day-2/)
- [Day 3](/tensorflow/2021/12/14/advent-of-code-tensorflow-day-3/)
- [Day 4](/tensorflow/2021/12/17/advent-of-code-tensorflow-day-4/)
- [Day 5](/tensorflow/2021/12/22/advent-of-code-tensorflow-day-5/)
- [Day 6](/tensorflow/2021/12/25/advent-of-code-tensorflow-day-6/)
- [Day 7](/tensorflow/2021/12/28/advent-of-code-tensorflow-day-7/)
- [Day 8](/tensorflow/2021/12/28/advent-of-code-tensorflow-day-8/)

The next article will be about my solution to Day 10 problem. The problem is completely different from the one faced so far, and it will be a nice showcase on how to use TensorFlow for working with strings (spoiler: the challenge is about text parsing and syntax checking!).

For any feedback or comment, please use the Disqus form below - thanks!
