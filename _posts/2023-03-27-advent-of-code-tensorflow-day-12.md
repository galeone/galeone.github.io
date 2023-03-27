---
layout: post
title: "Advent of Code 2022 in pure TensorFlow - Day 12"
date: 2023-03-27 08:00:00
categories: tensorflow
summary: "Solving problem 12 of the AoC 2022 in pure TensorFlow is a great exercise in graph theory and more specifically in using the Breadth-First Search (BFS) algorithm. This problem requires working with a grid of characters representing a graph, and the BFS algorithm allows us to traverse the graph in the most efficient way to solve the problem."
authors:
    - pgaleone
    - chatGPT
---

Solving problem 12 of the AoC 2022 in pure TensorFlow is a great exercise in graph theory and more specifically in using the Breadth-First Search (BFS) algorithm. This problem requires working with a grid of characters representing a graph, and the BFS algorithm allows us to traverse the graph in the most efficient way to solve the problem.

## [Day 12: Hill Climbing Algorithm](Day 12: Hill Climbing Algorithm)

You can click on the title above to read the full text of the puzzle. The TLDR version is: you are given a grid representing a maze, where each cell contains a letter from the English alphabet (lowercase), an 'S' to indicate the starting point, or an 'E' to indicate the ending point. The goal is to find the shortest path from the starting point to the ending point, following specific rules for navigating the maze.

Here's an example of the input grid:

```text
Sabc
defg
hEij
```

To move from the starting point to the ending point, you can only move to cells with the next letter in alphabetical order. In this case, the shortest path would be "S -> a -> b -> c -> d -> e -> f -> g -> h -> E", with a total of 9 steps.

NOTE: the goal is not to reach *precisely* the endpoint, you need to reach a point at the same elevation of `E` (in the input data, `z`, for the example above `h`).

Part 2 of this problem can be designed as the inverse problem: you start from the `E` point and you need to reach at point at the same elevation of `S` (thus, any possible `a` value in the grid) via the shortest path.

### Design Phase

The problem can be tackled using a Breadth-First Search (BFS) algorithm to traverse the graph represented by the input grid. The BFS algorithm is ideal for this task as it allows us to explore all possible paths in the most efficient way, ensuring that we find the shortest path.

We'll implement the BFS algorithm using TensorFlow's `tf.queue.FIFOQueue`  to maintain the order of the nodes to visit. In addition, we'll use a `visited` tensor to keep track of the cells we've already visited, which will help us avoid visiting the same cell multiple times and prevent infinite loops.

The provided Python code supports solving both part 1 and part 2 of the problem, with slight differences in the BFS traversal. The main difference between the two parts is the condition for moving from one cell to another. In part 1, you can only move to cells with the next letter in alphabetical order, while in part 2, you can move to cells with the previous letter in alphabetical order.

### Part 1 and Part 2 Solution

The code below contains the main function `main` that reads the input file and sets up the input. We first preprocess the input data by converting the characters to integers for easier processing. We create a lookup table to map characters to integers and apply this mapping to the dataset.

```python
dataset = tf.data.TextLineDataset(input_path.as_posix())
dataset = dataset.map(tf.strings.bytes_split)

keys_tensor = tf.concat(
    [tf.strings.bytes_split(string.ascii_lowercase), tf.constant(["S", "E"])],
    axis=0,
)
values_tensor = tf.concat([tf.range(0, 26), tf.constant([-1, 26])], axis=0)
lut = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor),
    default_value=-1,
)

dataset = dataset.map(lut.lookup)

grid = tf.convert_to_tensor(list(dataset))
```

The `grid` tensor now contains our 2D world. We can now go straight to the BFS implementation. Implementing the BFS algorithm requires just a simple data structure (a queue), and a support variable (`visited`) that we use to keep track of the already visited neighbors and, thus, avoid useless recomputions.

```python
queue = tf.queue.FIFOQueue(
    tf.cast(tf.reduce_prod(tf.shape(grid)), tf.int32),
    tf.int32,
    (3,),  # x,y,distance
)

visited = tf.Variable(tf.zeros_like(grid))
```

The `bfs` function is the core of our solution. This function takes an optional argument `part2` , which is set to `False` by default for solving part 1. To solve part 2, we simply call the function with `part2=True`.

The BFS algorithm starts by enqueuing the starting point (or the ending point for part 2) into the queue, along with an initial distance of 0. Then, while the queue is not empty, we dequeue the next cell to visit, along with its distance from the starting point. We then check if this cell has been visited before. If it has not been visited, we update the `visited`  tensor and check if the dequeued cell is the destination (either 'E' for part 1 or 'S' for part 2). If it is the destination, we return the distance as the shortest path length. Otherwise, we continue exploring the neighboring cells that satisfy the condition for traversal, depending on the part we are solving.

Of course, working on a 2D world we need to be able to move and "look around". We can thus define a `_neighs` function that given a point on the 2D grid, gives us the 4-neighbors.

```python
@tf.function
def _neighs(grid: tf.Tensor, center: tf.Tensor):
    y, x = center[0], center[1]

    shape = tf.shape(grid) - 1

    if tf.logical_and(tf.less(y, 1), tf.less(x, 1)):  # 0,0
        mask = tf.constant([(1, 0), (0, 1)])
    elif tf.logical_and(tf.equal(y, shape[0]), tf.equal(x, shape[1])):  # h,w
        mask = tf.constant([(-1, 0), (0, -1)])
    elif tf.logical_and(tf.less(y, 1), tf.equal(x, shape[1])):  # top right
        mask = tf.constant([(0, -1), (1, 0)])
    elif tf.logical_and(tf.less(x, 1), tf.equal(y, shape[0])):  # bottom left
        mask = tf.constant([(-1, 0), (0, 1)])
    elif tf.less(x, 1):  # left
        mask = tf.constant([(1, 0), (-1, 0), (0, 1)])
    elif tf.equal(x, shape[1]):  # right
        mask = tf.constant([(-1, 0), (1, 0), (0, -1)])
    elif tf.less(y, 1):  # top
        mask = tf.constant([(0, -1), (0, 1), (1, 0)])
    elif tf.equal(y, shape[0]):  # bottom
        mask = tf.constant([(0, -1), (0, 1), (-1, 0)])
    else:  # generic
        mask = tf.constant([(-1, 0), (0, -1), (1, 0), (0, 1)])

    coords = center + mask
    neighborhood = tf.gather_nd(grid, coords)
    return neighborhood, coords
```

The function is pretty borind to read: it handles all the cases in which the passed `center` parameter is a point along the border of the grid.

### Breadth-First Search using `tf.queue.FIFOQueue`

The key to our BFS implementation is the use of the tf.queue.FIFOQueue for maintaining the order of the nodes to visit. The FIFO (first-in, first-out) property ensures that we visit the nodes in the correct order, always visiting the closest nodes to the starting point first. This guarantees that we find the shortest path to the destination.

We initialize the queue with the starting point (or the ending point for part 2) and its distance from the starting point. While the queue is not empty, we dequeue the next cell to visit, along with its distance. We then check if the cell has been visited before and update the `visited` tensor accordingly. If the dequeued cell is the destination, we return the distance as the shortest path length. Otherwise, we enqueue the neighboring cells that satisfy the condition for traversal, along with their distances from the starting point.

```python
def bfs(part2=tf.constant(False)):
    if tf.logical_not(part2):
        start = tf.cast(tf.where(tf.equal(grid, -1))[0], tf.int32)
        queue.enqueue(tf.concat([start, tf.constant([0])], axis=0))
        dest_val = 25

        def condition(n_vals, me_val):
            return tf.where(tf.less_equal(n_vals, me_val + 1))

    else:
        end = tf.cast(tf.where(tf.equal(grid, 26)), tf.int32)[0]
        queue.enqueue(tf.concat([end, tf.constant([0])], axis=0))
        dest_val = 1

        def condition(n_vals, me_val):
            return tf.where(tf.greater_equal(n_vals, me_val - 1))

    while tf.greater(queue.size(), 0):
        v = queue.dequeue()
        me, distance = v[:2], v[2]
        me_val = tf.gather_nd(grid, [me])
        already_visited = tf.squeeze(tf.cast(tf.gather_nd(visited, [me]), tf.bool))
        if tf.logical_not(already_visited):
            if tf.reduce_all(tf.equal(me_val, dest_val)):
                return distance - 1
            visited.assign(tf.tensor_scatter_nd_add(visited, [me], [1]))

            n_vals, n_coords = _neighs(grid, me)
            potential_dests = tf.gather_nd(
                n_coords,
                condition(n_vals, me_val),
            )

            not_visited = tf.equal(tf.gather_nd(visited, potential_dests), 0)
            neigh_not_visited = tf.gather_nd(potential_dests, tf.where(not_visited))

            to_visit = tf.concat(
                [
                    neigh_not_visited,
                    tf.reshape(
                        tf.repeat(distance + 1, tf.shape(neigh_not_visited)[0]),
                        (-1, 1),
                    ),
                ],
                axis=1,
            )
            queue.enqueue_many(to_visit)

    return -1
```

The use of `tf.queue.FIFOQueue` in our BFS implementation allows us to efficiently explore the graph while maintaining the traversal order, enabling us to find the shortest path between the starting point and the destination.

Finally, we call the `bfs` function for both parts, reset the queue and visited tensor in between, and print the results.

```python
tf.print("Steps: ", bfs())
queue.dequeue_many(queue.size())
visited.assign(tf.zeros_like(visited))

tf.print("Part 2: ", bfs(True))
```

## Conclusion

You can the solution in folder `12` in the dedicated GitHub repository (in the `2022` folder): [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

In this article, we have demonstrated how to solve problem 12 of the AoC 2022 using TensorFlow, focusing on the implementation of the Breadth-First Search algorithm with `tf.queue.FIFOQueue`. We have also shown how the provided Python code supports solving both part 1 and part 2 of the problem, highlighting the differences between the two parts. The BFS algorithm, along with the use of TensorFlow's `tf.queue.FIFOQueue`, provides an efficient and elegant solution to this graph traversal problem.

If you missed the article about the previous days’ solutions, here's a handy list

- [Advent of Code 2022 in pure TensorFlow - Days 1 & 2](/tensorflow/2022/12/04/advent-of-code-tensorflow-day-1-and-2/).
- [Advent of Code 2022 in pure TensorFlow - Days 3 & 4](/tensorflow/2022/12/11/advent-of-code-tensorflow-day-3-and-4/).
- [Advent of Code 2022 in pure TensorFlow - Day 5](/tensorflow/2022/12/21/advent-of-code-tensorflow-day-5/)
- [Advent of Code 2022 in pure TensorFlow - Day 6](/tensorflow/2022/12/27/advent-of-code-tensorflow-day-6/)
- [Advent of Code 2022 in pure TensorFlow - Day 7](/tensorflow/2022/12/29/advent-of-code-tensorflow-day-7/)
- [Advent of Code 2022 in pure TensorFlow - Day 8](/tensorflow/2023/01/14/advent-of-code-tensorflow-day-8/)
- [Advent of Code 2022 in pure TensorFlow - Day 9](/tensorflow/2023/01/23/advent-of-code-tensorflow-day-9/)
- [Advent of Code 2022 in pure TensorFlow - Day 10](/tensorflow/2023/03/25/advent-of-code-tensorflow-day-10/)
- [Advent of Code 2022 in pure TensorFlow - Day 11](/tensorflow/2023/03/26/advent-of-code-tensorflow-day-11/)

For any feedback or comment, please use the Disqus form below - thanks!
