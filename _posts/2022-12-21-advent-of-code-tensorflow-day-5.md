---
layout: post
title: "Advent of Code 2022 in pure TensorFlow - Day 5"
date: 2022-12-21 05:00:00
categories: tensorflow
summary: "In the first part of the article, I'll explain the solution that solves completely both parts of the puzzle. As usual, focusing on the TensorFlow features used during the solution and all the various technical details worth explaining. In the second part, instead, I'll propose a potential alternative solution to the problem that uses a tf.Variable with an undefined shape. This is a feature of tf.Variable that's not clearly documented and, thus, widely used. So, at the end of this article, we'll understand how to solve the day 5 problem in pure TensorFlow and also have an idea of how to re-design the solution using a tf.Variable with the validate_shape argument set to False."
authors:
    - pgaleone
---

Differently from the previous 2 articles, where I merged the description of the solutions of two problems into one article, this time the whole article is dedicated to the pure TensorFlow solution to problem number 5. The reason is simple: solving this problem in pure TensorFlow hasn't been straightforward so it is worth explaining all the limitations and the subtleties found during the solution.

In the first part of the article, I'll explain the solution that solves completely both parts of the puzzle. As usual, focusing on the TensorFlow features used during the solution and all the various technical details worth explaining. In the second part, instead, I'll propose a potential alternative solution to the problem that uses a `tf.Variable` with an "undefined shape". This is a feature of `tf.Variable` that's not clearly documented and, thus, widely used. So, at the end of this article, we'll understand how to solve the day 5 problem in pure TensorFlow and also have an idea of how to re-design the solution using a `tf.Variable` with the `validate_shape` argument set to `False`.

## [Day 5: Supply Stacks](https://adventofcode.com/2022/day/5)

You can click on the title above to read the full text of the puzzle. The TLDR version is: we have an initial configuration of crates (the puzzle input) and a set of moves to perform. Part 1 constrains the crane that's moving the crates to pick a single crate at a time, while part 2 allows multiple crates to be picked up at the same time. The problem asks us to determine, after having moved the crates, what crate ends up on top of each stack.

So, given a puzzle input like
```
    [D]
[N] [C]
[Z] [M] [P]
 1   2   3

move 1 from 2 to 1
move 3 from 1 to 3
move 2 from 2 to 1
move 1 from 1 to 2
```

Part 1, which wants us to move the crates one at a time, ends up with this final configuration

<pre>
        [<b>Z</b>]
        [N]
        [D]
[<b>C</b>] [<b>M</b>] [P]
 1   2   3
 </pre>

Thus, the result is: "CMZ".

For part 2 instead, where multiple crates can be picked up at the same time, the final configuration is

<pre>
        [<b>D</b>]
        [N]
        [Z]
[<b>M</b>] [<b>C</b>] [P]
 1   2   3
 </pre>

in this case the result is "MCD".

### Design Phase

The problem can be breakdown into 4 simple steps:

1. Read the first part of the input: parse the crates
1. Create a data structure that models the stacks of crates
1. Read the second part of the input: parse the instructions
1. Iteratively transform the previously created data structure according to the instructions


### Parsing strings: tf.strings & regex

The [`tf.strings`](https://www.tensorflow.org/api_docs/python/tf/strings/) module contains several utilities for working with `tf.Tensor` with `dtype=tf.string`. Unfortunately, there are **tons** of limitations when working with strings and `tf.function`-decorated functions (we'll see some of these limitations later). Moreover, perhaps the most powerful tool for string parsing and manipulation (the [regular expressions](https://en.wikipedia.org/wiki/Regular_expression)) has a very limited integration. We only have 2 functions to use:

- [tf.strings.regex_full_match](https://www.tensorflow.org/api_docs/python/tf/strings/regex_full_match): check if the input matches the regex pattern.
- [tf.strings.regex_replace](https://www.tensorflow.org/api_docs/python/tf/strings/regex_replace(input, pattern, rewrite)): replace elements of input matching regex pattern with rewrite.

That's all. So, if we want to use regular expressions for parsing the crates or the moves, we are forced to use only these 2 functions, together with the other basic string manipulation functions offered by the module.


### Parsing the crates

We are quite lucky because every line containing at least a crate contains the character `[`. Thus, a regex pattern for this pattern is `".*\[.*`.

We can thus easily filter every line that contains a crate by applying this condition to every line of the input dataset.

```python
stacks_dataset = dataset.filter(
        lambda line: tf.strings.regex_full_match(line, r".*\[.*")
    )
```

Now, we need to find a way to extract every stack of crates. This can be quite easily done noticing that every stack is just a column of 4 characters. Thus, by moving 4 characters at a time over every line, we can understand where a crate is and extract its letter.

Of course, the tool to use is `tf.TensorArray` since every stack could contain a variable number of crates, and `tf.TensorArray` is the only tool we can use in static-graph mode for having such behavior.


```python
@tf.function
def to_array(line):
    length = tf.strings.length(line) + 1
    stacks = length // 4
    ta = tf.TensorArray(tf.string, size=0, dynamic_size=True)
    for i in tf.range(stacks):
        substr = tf.strings.strip(tf.strings.substr(line, i * 4, 4))
        stripped = tf.strings.regex_replace(substr, r"\[|\]", "")
        ta = ta.write(i, stripped)

    return ta.stack()
```

the `to_array` function accepts a line `[Z] [M] [P]` as input, parses it by moving 4 characters at a time along its length, removes the square brackets, and saves the letter in the correct position of the stack (0 indexed).

```python
stacks_dataset = stacks_dataset.map(to_array)
```

The lines are read from the first to the last, that means every stack is built from the *top to the bottom*.

Now that we have a dataset that produces stacks when iterated, we can convert it to a `tf.Tensor`. In this way our `stacks_tensor` contains a bunch of very important information:

- The number of stacks (the shape along the 1-dimension)
- The height of the highest stack


```python
stacks_tensor = tf.convert_to_tensor(list(stacks_dataset))
num_stacks = tf.shape(stacks_tensor, tf.int64)[1] + 1
```

`stacks_tensor` is unmodifiable not being a `tf.Variable`. However, we are in a particular situation in which TensorFlow is not perfectly suited. In fact, even if `stacks_tensor' is converted into a `tf.Variable` we have no idea about the maximum height that can be reached when applying the various instructions and moving the crates.

### Stacking crates: the limitations of tf.Variable

As it's easy to understand, we have no idea how tall a stack can become, everything depends on the instructions. Since a `tf.Variable` **requires** a well-defined shape, we are forced to guess this maximum size and hope it's enough. That's for sure a huge limitation of this solution.

However, in the last part of the article I'll suggest to the reader a potential starting point for designing a more general solution that uses the `validate_shape=False` parameter of the `tf.Variable` - that's a not widely used feature, but very powerful.

```python
max_stack_size = 200
stacks = tf.Variable(tf.zeros((max_stack_size, num_stacks - 1, 1), dtype=tf.string))
```

`stacks` is a `tf.Variable` with shape `(200, number of stacks -1, 1)`:

- 200 is the maximum number of crates in a stack our solution supports (the limitation)
- `num_stacks -1` is the number of stacks (0-based).
- `1` is the dimension of the inner dimension. Let's say that's not required (since the unary dimensions can be squeezed) but having it simplifies the usage of the `tf.tensor_scatter_nd_*` functions.

The `stacks` variable is initialized with zeros, but we have our `stacks_tensor` that represents the initial state. So, we can define a function that initializes to the initial state the `stacks` variable every time is called.

```python
def initialize_stacks():
    stacks.assign(tf.zeros_like(stacks))

    indices_x, indices_y = tf.meshgrid(
        tf.range(max_stack_size - tf.shape(stacks_tensor)[0], max_stack_size),
        tf.range(tf.shape(stacks_tensor)[1]),
    )
    indices = tf.stack([indices_x, indices_y], axis=-1)

    updates = tf.expand_dims(tf.transpose(stacks_tensor), axis=2)
    stacks.assign(tf.tensor_scatter_nd_update(stacks, indices, updates))

initialize_stacks()
```

First, we reset the status of the `stacks` variable to zero. Then, we create the indices we want to use while assigning elements of `updates` to `stacks`. The idea is to assign the elements "at the bottom" which means that a stack in `stack_tensor` (like `ZND` up to down) is assigned to its corresponding stack in the `stacks` tensor at the position `max_stack_size -3` onwards (e.g. `Z` in position `max_stack_size-3`, `N` in position max_stack_size-2`, and so on).

Then, we use the [`tf.tensor_scatter_nd_update`](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update) function to scatter `updates` into `stacks` according to `indices`. This function returns a new tensor with the same shape of `stacks`, that thus can be perfectly assigned to the `stacks` variable.

Having a single `tf.Variable` containing all the stacks at once, we have lost the information about the number of crates per stack. In fact, `tf.shape(stacks)` will always return the whole variable shape without information about the non-empty elements present along every dimension.

Thus, we need to keep track manually of this information, which needs to be re-computed every time we apply a transformation to the `stacks` variable.


### Counting crates: mutable hashmaps

The idea is simple: the zero element for the  `tf.string` dtype is the empty string (`""`). Thus, we can count the number of non-empty elements along the 0-axis, and save the result inside a mutable hashmap that maps the stack number to its number of crates.

Once again, the perfect tool (although still experimental, it works fine!) is [tf.lookup.experimental.MutableHashTable](https://www.tensorflow.org/api_docs/python/tf/lookup/experimental/MutableHashTable).

```python
num_elements = tf.lookup.experimental.MutableHashTable(
    tf.int64, tf.int64, default_value=-1
)

def update_num_elements():
    num_elements.insert(
        tf.range(num_stacks - 1),
        tf.squeeze(
            tf.reduce_sum(tf.cast(tf.not_equal(stacks, ""), tf.int64), axis=[0])
        ),
    )

update_num_elements()
```

Here we go, we have the `stacks` variable populated with the initial state, and the `num_elements` lookup table that contains the number of elements for each stack.

We can now parse the remaining part of the dataset and, line by line (thus, while looping over the dataset), execute all the instructions provided.

### Instruction parsing & execution

Part 1 requires to move one crate at a time, while part 2 requires to move the specified number of creates at the same time.

Moving one element at a time, means inserting the elements in reversed order (read `ABC`, insert `CBA`). We can keep track of this requirement with a boolean variable.

```python
one_at_a_time = tf.Variable(True)
```

Now we can focus on the line parsing. As anticipated, we have a very limited number of functions in the `tf.strings` package for working with strings, thus the manipulation is a bit cumbersome.

Given a line in the format

```
move X from A to B
```

our goal is to extract the numbers `X` (amount), `A` (source), `B` (destination) and use them for reading from the stack with ID `source`, the required `amount` of elements from the top (where our "top" is given by the number of nonempty elements in the stack), and move them into `B` according to the strategy defined by `one_at_a_time`.


```python
def move(line):
    amount = tf.strings.to_number(
        tf.strings.regex_replace(
            tf.strings.regex_replace(line, "move ", ""), r" from \d* to \d*$", ""
        ),
        tf.int64,
    )

    source_dest = tf.strings.regex_replace(line, r"move \d* from ", "")
    source = (
        tf.strings.to_number(
            tf.strings.regex_replace(source_dest, r" to \d*$", ""), tf.int64
        )
        - 1
    )

    dest = (
        tf.strings.to_number(
            tf.strings.regex_replace(source_dest, r"\d* to ", ""), tf.int64
        )
        - 1
    )

    num_element_source = num_elements.lookup([source])[0]
    top = max_stack_size - num_element_source

    read = stacks[top : top + amount, source]

    # remove from source
    indices_x, indices_y = tf.meshgrid(tf.range(top, top + amount), [source])
    indices = tf.reshape(tf.stack([indices_x, indices_y], axis=-1), (-1, 2))
    updates = tf.reshape(tf.repeat("", amount), (-1, 1))

    stacks.assign(
        tf.tensor_scatter_nd_update(stacks, indices, updates), use_locking=True
    )

    num_element_dest = num_elements.lookup([dest])[0]
    top = max_stack_size - num_element_dest - 1

    # one a at a time -> reverse
    if one_at_a_time:
        insert = tf.reverse(read, axis=[0])
        insert = tf.reshape(insert, (-1, 1))
    else:
        insert = tf.reshape(read, (-1, 1))

    indices_x, indices_y = tf.meshgrid(tf.range(top - amount + 1, top + 1), [dest])
    indices = tf.reshape(tf.stack([indices_x, indices_y], axis=-1), (-1, 2))

    stacks.assign(
        tf.tensor_scatter_nd_update(stacks, indices, insert), use_locking=True
    )

    update_num_elements()
    return stacks
```

The `move` function does all the job. Reads the line, parses it, extracts the values, removes them from source (set the to empty string), and inserts into the destination stack the read values according to the strategy.

It's not immediate to understand all the indices manipulation, so take your time for reading the code carefully. In particular, the reader should focus on the [`tf.meshgrid`](https://www.tensorflow.org/api_docs/python/tf/meshgrid?hl=en) function used to create the various indices along the two dimensions.


### Solving the problem

The `move` function works only on the instructions lines, thus we need to create a dataset that only contains these lines. It's trivial with the `tf.data.Dataset.skip` method.

```python
moves_dataset = dataset.skip(tf.shape(stacks_tensor, tf.int64)[0] + 2)
```

We are now ready to play, solve the problem and visualize the final stack:


```python
tf.print("part 1")
play = moves_dataset.map(move)

list(play)

indices_x = tf.range(num_stacks - 1)
indices_y = max_stack_size - tf.reverse(num_elements.export()[1], axis=[0])
indices = tf.reshape(tf.stack([indices_y, indices_x], axis=-1), (-1, 2))

tf.print(tf.strings.join(tf.squeeze(tf.gather_nd(stacks, indices)), ""))
```

The thing worth noting is that playing this game is the iteration of the `play` dataset, invocated by converting the dataset object to a list (`list(play)`).

The `num_elements` lookup table contains the index of every top-crate, thus we can just gather them, join as a single string and print them for getting the expected result.

Part two is identical, we only need to toggle the `one_at_a_time` variable, reset the `stacks` variable to its initial state, play once again the very same game, and gather the result as above.

```python
tf.print("part 2")
initialize_stacks()
update_num_elements()
one_at_a_time.assign(False)
play = moves_dataset.map(move)
list(play)

indices_x = tf.range(num_stacks - 1)
indices_y = max_stack_size - tf.reverse(num_elements.export()[1], axis=[0])
indices = tf.reshape(tf.stack([indices_y, indices_x], axis=-1), (-1, 2))

tf.print(tf.strings.join(tf.squeeze(tf.gather_nd(stacks, indices)), ""))
```

Here we go, day 5 problem solved in pure TensorFlow!


## tf.Variable undefined shape & validation

In this part of the article, I just want to highlight a not widely used behavior of the `tf.Variable` object: the possibility of creating a `tf.Variable` without specifying its shape.

The `tf.Variable` [documentation](https://www.tensorflow.org/api_docs/python/tf/Variable?hl=en) mentions the `validate_shape` parameter only twice:

1. When documenting the `initial_value` parameter.
   > The initial value must have a shape specified unless validate_shape is set to False.
1. When documenting the `validate_shape` parameter itself.
   > If False, allows the variable to be initialized with a value of unknown shape. If True, the default, the shape of `initial_value` must be known.

So, by reading the documentation we can say that's possible to define a `tf.Variable` without shape, assigning it a `tf.Tensor` with a different shape without any problem. In this way, we could avoid the limitation of specifying a `max_stack_size` and writing a very general solution without this hard constraint.

So, instead of giving a complete solution I just want to leave here a potential starting point for designing a more general solution.


```python
stacks = tf.Variable(
   stacks_tensor, validate_shape=False, dtype=tf.string, shape=tf.TensorShape(None)
)
# ...

def initialize_stacks():
    stacks.assign(tf.zeros_like(stacks))
    shape = tf.shape(stacks)
    stacks.assign(
       tf.reshape(
           stacks,
           [
               shape[0],
               shape[1],
               1,
           ],
       )
    )
    # ...
```

The goal is the remove from the [solution](https://github.com/galeone/tf-aoc/blob/main/2022/5/main.py) the `max_stack_size` and always work with a `tf.Variable` with a variable shape: the tricky part will be the indexing, I guarantee it!

## Conclusion

You can see the complete solution in folder `5` in the dedicated GitHub repository (in the `2022` folder): [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

Using TensorFlow for solving this problem allowed us to understand that the `tf.strings` package contains only a small set of utilities for working with strings, and that the regular expression support is quite limited.
Once again, we relied upon the *experimental* `tf.lookup.experimental.MutableHashTable`, although still experimental (by several years!) it works quite well.

In the last part of the article I suggested an alternative approach to the problem, that should allow a better design of this solution without a constraint on the maximum number of stacks (and thus, without a constraint on the variable dimension). If you want to contribute and submit to the repository a merge request with your alternate solution in pure TensorFlow with a `tf.Variable` with an undefined shape, I'd be very happy to review and talk about it!

If you missed the article about the previous days’ solutions, here's a handy list

- [Advent of Code 2022 in pure TensorFlow - Days 1 & 2](/tensorflow/2022/12/04/advent-of-code-tensorflow-day-1-and-2/).
- [Advent of Code 2022 in pure TensorFlow - Days 3 & 4](/tensorflow/2022/12/11/advent-of-code-tensorflow-day-3-and-4/).

For any feedback or comment, please use the Disqus form below - thanks!
