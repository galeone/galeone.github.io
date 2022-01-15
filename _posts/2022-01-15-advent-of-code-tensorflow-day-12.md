---
layout: post
title: "Advent of Code 2021 in pure TensorFlow - day 12"
date: 2022-01-15 08:00:00
categories: tensorflow
summary: "Day 12 problem projects us the world of graphs. TensorFlow can be used to work on graphs pretty easily since a graph can be represented as an adjacency matrix, and thus, we can have a tf.Tensor containing our graph. However, the \"natural\" way of exploring a graph is using recursion, and as we'll see in this article, this prevents us to solve the problem using a pure TensorFlow program, but we have to work only in eager mode. "
authors:
    - pgaleone
---

Day 12 problem projects us the world of graphs. TensorFlow can be used to work on graphs pretty easily since a graph can be represented as an adjacency matrix, and thus, we can have a tf.Tensor containing our graph. However, the "natural" way of exploring a graph is using recursion, and as we'll see in this article, this prevents us to solve the problem using a pure TensorFlow program, but we have to work only in eager mode.


## [Day 12: Passage Pathing](https://adventofcode.com/2021/day/12)

You can click on the title above to read the full text of the puzzle. The TLDR version is:

Our dataset is a list of connections with this format:

```
start-A
start-b
A-c
A-b
b-d
A-end
b-end
```

where `start` and `end` are your start and end points, the upper case letters (e.g `A`) are the "big caves", and the lower case letters (e.g. `c`) are the "small caves".

The above cave system looks roughly like this:

```
    start
    /   \
c--A-----b--d
    \   /
     end
```

The puzzle asks to **find the number of distinct paths** that start at `start`, and end at `end`, knowing that you can visit the "big caves" multiple times, and the small caves only once (per path).

Given these rules, there are 10 paths through this example cave system (graph)

```
start,A,b,A,c,A,end
start,A,b,A,end
start,A,b,end
start,A,c,A,b,A,end
start,A,c,A,b,end
start,A,c,A,end
start,A,end
start,b,A,c,A,end
start,b,A,end
start,b,end
```

### Design phase: part one

The problem asks us to:

1. Model the relationships between the nodes
2. Finding the paths from `start` to `end` that follow the two rules about the big/small caves

Modeling the relationships between nodes is something we can achieve by mapping the neighboring relationship among nodes using an [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix). In practice, we can assign an unique index to a node (e.g. "start" = 0, "end" = 1, "A" = 2, ...), and each index identifies a row/column in the adjacency matrix.
Whenever a node is connected to another, we put a "1" in the correct location.

For example, a possible adjacency matrix for the example graph is

$$
\begin{pmatrix}
0&1&1&0&0&0\\
1&0&1&1&0&1\\
1&1&0&0&1&1\\
0&1&0&0&0&0\\
0&0&1&0&0&0\\
0&1&1&0&0&0\\
\end{pmatrix}
$$

As it can be easily seen the adjacency matrix for a graph without self connections is a [hollow matrix](https://en.wikipedia.org/wiki/Hollow_matrix) that means that all the elements along the diagonal are equal to zero. Moreover, the adjacency matrix is also symmetric (rows and columns can be transposed and the matrix doesn't change - and it makes sense since if `A` is connected with `B` also `B` is connected with `A`).

Thus, we first need to create the mapping between the "human-readable" nodes and the IDs, then characterize these IDs. In fact, we have some rules to follow. All the upper case nodes (and thus the corresponding numeric IDs) can be visited only once, while the other can be visited multiple times.

Moreover, the `start` and `end` nodes are special, since if we reach the `end` the path is complete and we can exit from our search algorithm, while the `start` node shouldn't be visited more than once (at the beginning and then ignored).

### Input pipeline

We create a `tf.data.Dataset` object for reading the text file line-by-line [as usual](/tensorflow/2021/12/11/advent-of-code-tensorflow/#input-pipeline). We split every line looking at the `-` separator, and then create the relationships between the human-readable nodes and the indexes of the adjacency matrix

```python
connections = tf.data.TextLineDataset("input").map(
    lambda string: tf.strings.split(string, "-")
)

# Create a map between human-readable node names and numeric indices
human_to_id = tf.lookup.experimental.MutableHashTable(tf.string, tf.int64, -1)
id_to_human = tf.lookup.experimental.MutableHashTable(tf.int64, tf.string, "")
```

`human_to_id` is the hashtable we use to create the mapping between the human-readable node name and a numeric ID. [`tf.lookup.experimental.MutableHashTable`](https://www.tensorflow.org/api_docs/python/tf/lookup/experimental/MutableHashTable) returns the third parameter (-1) when the lookup fails, and thus we can use this feature to check if we already created the mapping for a node.


### Adjacency matrix

Since we want to create an adjacency matrix, we need to keep track of the adjacency relations between nodes (the `indices` we use for setting the `1` in the correct positions of the adjacency matrix), we use a `tf.TensorArray` for doing it.

```python
idx = tf.Variable(0, dtype=tf.int64)
indices = tf.TensorArray(tf.int64, size=0, dynamic_size=True)
for edge in connections:
    node_i = human_to_id.lookup(edge[0])
    node_j = human_to_id.lookup(edge[1])

    if tf.equal(node_i, -1):
        human_to_id.insert([edge[0]], [idx])
        id_to_human.insert([idx], [edge[0]])
        node_i = tf.identity(idx)
        idx.assign_add(1)
    if tf.equal(node_j, -1):
        human_to_id.insert([edge[1]], [idx])
        id_to_human.insert([idx], [edge[1]])
        node_j = tf.identity(idx)
        idx.assign_add(1)

    ij = tf.convert_to_tensor([node_i, node_j])
    indices = indices.write(indices.size(), ij)
```

We loop over all the connections, check if the node has already been mapped and, if not, increase the counter (`idx`) and then create the mapping. Then, we create the relationship between the `i` and `j` nodes and save it inside the `tf.TensorArray`.

At the end of this loop, the `indices` variable contains all the pairs of coordinates where a connection is, while `idx` contains the ID of the latest mapped item, which also is the width (height) of the matrix.

We should also note that we created the relationship from `i` to `j`, hence we created an **upper triangular matrix** (a matrix whose elements below the main diagonals are zero).
For modeling the full relationship `i` to `j` **and** `j` to `i` we can just transpose the matrix and sum it with itself.

```python
indices = indices.stack()
indices = tf.reshape(indices, (-1, 2))
A = tf.tensor_scatter_nd_update(
    tf.zeros((idx, idx), dtype=tf.int64),
    indices,
    tf.repeat(tf.cast(1, tf.int64), tf.shape(indices)[0]),
)
A = A + tf.transpose(A)
```

Here we go! `A` is the adjacency matrix that completely models the graph.

The only thing missing is the modeling of the constraints. We need a list-like object (a `tf.Tensor`) containing all the IDs of the nodes that we can visit once, and the difference between this set of IDs and the full list of IDs for getting the list of the IDs we can visit multiple times.

```python
keys = human_to_id.export()[0]
visit_only_once_human = tf.gather(
    keys,
    tf.where(
        tf.equal(
            tf.range(tf.shape(keys)[0]),
            tf.cast(tf.strings.regex_full_match(keys, "[a-z]+?"), tf.int32)
            * tf.range(tf.shape(keys)[0]),
        )
    ),
)
visit_only_once_human = tf.squeeze(visit_only_once_human)
visit_only_once_id = human_to_id.lookup(visit_only_once_human)

# Visit multiple times = {keys} - {only once}
visit_multiple_times_human = tf.sparse.to_dense(
    tf.sets.difference(
        tf.reshape(keys, (1, -1)), tf.reshape(visit_only_once_human, (1, -1))
    )
)
visit_multiple_times_human = tf.squeeze(visit_multiple_times_human)
visit_multiple_times_id = human_to_id.lookup(visit_multiple_times_human)
```

It's interesting how we used **regular expressions** in TensorFlow with the `tf.strings.regex_full_match` function for searching for nodes all in lower case (`[a-z]+?` - visit only once).

Moreover, we used the [set difference](https://www.tensorflow.org/api_docs/python/tf/sets/difference) together with the conversion [from sparse to dense](https://www.tensorflow.org/api_docs/python/tf/sparse/to_dense?hl=en) to get a `tf.Tensor` containing all the upper-case nodes (visit multiple times).

### Visiting the graph

The natural way of facing this problem is by using **recursion**. In fact, the problem can be modeled as follows

> Concatenate the input `node` to the `current_path`.
> 
> Is the current node the `end` node? If yes, we found the path. Increment by 1 the counter of the paths. Return.
>
> Otherwise, find the neighborhood of the current node.
>
> For every neighbor verify if the neighbor is in the "visit only once" list and it's in the current path. If yes, skip this neighbor.
>
> Otherwise, visit the neighbor (recursive step).

Being the adjacency matrix symmetric, the step of "find the neighborhood of the current node" can be simply modeled as a search for the `1` in the row whose id is `node_id`.

```python
@tf.function
def _neigh_ids(A, node_id):
    return tf.squeeze(tf.where(tf.equal(A[node_id, :], 1)))
```

Since the proposed solution uses the recursion we  **cannot** decorate the `visit` function with `@tf.function` because [tf.function does not support recursion](https://github.com/tensorflow/tensorflow/issues/35540#:~:text=tf.function%20does%20not%20support%20recursion). Thus, we are forced to solve this problem in eager mode.

Going back to the problem, the puzzle asks us to count the number of paths hence we need a `tf.Variable` for counting the number of paths found. Moreover, instead of repeating on every iteration a lookup in the hastable, we can extract the IDs of the `start` and `end` nodes.

```python
start_id, end_id = human_to_id.lookup(["start", "end"])
count = tf.Variable(0, dtype=tf.int64)
```

The visit function can be defined by implementing precisely what has been proposed in the algorithm described above.

```python
def _visit(A: tf.Tensor, node_id: tf.Tensor, path: tf.Tensor):
    current_path = tf.concat([path, [node_id]], axis=0)
    if tf.equal(node_id, end_id):
        count.assign_add(1)
        return current_path

    neighs = _neigh_ids(A, node_id)
    neigh_shape = tf.shape(neighs)
    if tf.equal(tf.size(neighs), 0):
        return current_path

    if tf.equal(tf.size(neigh_shape), 0):
        neighs = tf.expand_dims(neighs, 0)
        neigh_shape = tf.shape(neighs)

    for idx in tf.range(neigh_shape[0]):
        neigh_id = neighs[idx]
        if tf.logical_and(
            tf.reduce_any(tf.equal(neigh_id, visit_only_once_id)),
            tf.reduce_any(tf.equal(neigh_id, current_path)),
        ):
            continue
        # Recursion step
        _visit(A, neigh_id, current_path)
    return current_path
```

### Execution

To call the `_visit` function we need:

- `A` the adjacency matrix
- `node_id` a node for starting the visit
- `path` the initial path (a `tf.Tensor` containing only the ID of the `start` node).

```python
# All the paths starts from start
neighs = _neigh_ids(A, start_id)
for idx in tf.range(tf.shape(neighs)[0]):
    neigh_id = neighs[idx]
    _visit(A, neigh_id, [start_id])

tf.print("Part one: ", count)
```

Here we go! Part one is solved! Let's see what part two is about.

## Design phase: part 2

The puzzle relaxes a constraint: in a path, we can pass twice for a single small cave. For example `start,A,b,A,b,A,c,A,end` is a valid path because we visited `b` twice and `c` once.

Given this relaxed constraint, how many paths through the graph are there?

### Part two implementation

It is possible to simply add the new constraint to the previous algorithm. We can define an `inner_count` `tf.Variable` for counting the number of times small cave in the current path has been visited. When this counter is greater or equal to `2` then the path we are following is invalid and we can return it.

```python
count.assign(0)
inner_count = tf.Variable(0)

def _visit2(A: tf.Tensor, node_id: tf.Tensor, path: tf.Tensor):
    current_path = tf.concat([path, [node_id]], axis=0)

    # Skip start
    if tf.equal(node_id, start_id):
        return current_path

    # Success on end node
    if tf.equal(node_id, end_id):
        # paths.append(current_path)
        count.assign_add(1)
        return current_path

    # More than 2 lowercase visited twice
    visited, visited_idx, visited_count = tf.unique_with_counts(current_path)
    visited = tf.gather_nd(visited, tf.where(tf.greater(visited_count, 1)))
    inner_count.assign(0)
    for idx in tf.range(tf.shape(visited)[0]):
        if tf.reduce_any(tf.equal(visited[idx], visit_only_once_id)):
            inner_count.assign_add(1)

        if tf.greater_equal(inner_count, 2):
            return current_path

    neighs = _neigh_ids(A, node_id)
    neigh_shape = tf.shape(neighs)
    if tf.equal(tf.size(neighs), 0):
        return current_path

    if tf.equal(tf.size(neigh_shape), 0):
        neighs = tf.expand_dims(neighs, 0)
        neigh_shape = tf.shape(neighs)

    for idx in tf.range(neigh_shape[0]):
        neigh_id = neighs[idx]

        # already visited twice and is lowcase
        if tf.logical_and(
            tf.reduce_any(tf.equal(neigh_id, visit_only_once_id)),
            tf.greater(
                tf.reduce_sum(tf.cast(tf.equal(neigh_id, current_path), tf.int32)),
                1,
            ),
        ):
            continue

        _visit2(A, neigh_id, current_path)

    return current_path
```

The execution is performed in the very same way:

```python
neighs = _neigh_ids(A, start_id)
for idx in tf.range(tf.shape(neighs)[0]):
    neigh_id = neighs[idx]
    _visit2(A, neigh_id, [start_id])

tf.print("Part two: ", count)
```

It takes **a lot** of time since we are effectively doing the graph visit, even if the puzzle asked us to just **count** the number of paths (without modeling them!). Hence for sure, there exists a better solution, very similar to the solution I found to the [Day 6 puzzle part two](/tensorflow/2021/12/25/advent-of-code-tensorflow-day-6/#design-phase-part-2), but for this time the slow solution will be good enough. In fact, after about 20 minutes of computations the output produced is corrected :)

## Conclusion

You can see the complete solution in folder `12` on the dedicated Github repository: [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

Solving this problem in TensorFlow demonstrated how the adjacency matrix representation is really handy while working with graphs, but also demonstrated that is **impossible** to write recursive pure-TensorFlow programs (with `tf.function).

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
- [Day 11](/tensorflow/2022/01/08/advent-of-code-tensorflow-day-11/)

I already solved also part 1 of day 13, but I don't know if I have the time for solving the puzzles and writing the articles (holidays have reached the end some week ago :D).

So maybe, for the 2021 Advent of Code in pure TensorFlow is all.

For any feedback or comment, please use the Disqus form below - thanks!
