---
layout: post
title: "Advent of Code 2022 in pure TensorFlow - Day 7"
date: 2022-12-29 08:00:00
categories: tensorflow
summary: "Solving problem 7 of the AoC 2022 in pure TensorFlow allows us to understand certain limitations of the framework. This problem requires a lot of string manipulation, and TensorFlow (especially in graph mode) is not only not easy to use when working with this data type, but also it has a set of limitations I'll present in the article. Additionally, the strings to work with in problem 7 are (Unix) paths. TensorFlow has zero support for working with paths, and thus for simplifying a part of the solution, I resorted to the pathlib Python module, thus not designing a completely pure TensorFlow solution."
authors:
    - pgaleone
---

Solving problem 7 of the AoC 2022 in pure TensorFlow allows us to understand certain limitations of the framework. This problem requires a lot of string manipulation, and TensorFlow (especially in graph mode) is not only not easy to use when working with this data type, but also it has a set of limitations I'll present in the article. Additionally, the strings to work with in problem 7 are (Unix) paths. TensorFlow has zero support for working with paths, and thus for simplifying a part of the solution, I resorted to the `pathlib` Python module, thus not designing a completely pure TensorFlow solution.

## [Day 7: No Space Left On Device](https://adventofcode.com/2022/day/7)

You can click on the title above to read the full text of the puzzle. The TLDR version is: we are given a terminal output, containing some commands and the results of the execution of these commands. The commands are two standard Unix commands:

- `cd`: for changing directory, with support for full paths and relative paths
- `ls`: for listing the content of the directory I'm currently in.

The `ls` output contains information about the size of the type of the various files inside of the current working directory.

So, given a puzzle input like
```
$ cd /
$ ls
dir a
14848514 b.txt
8504156 c.dat
dir d
$ cd a
$ ls
dir e
29116 f
2557 g
62596 h.lst
$ cd e
$ ls
584 i
$ cd ..
$ cd ..
$ cd d
$ ls
4060174 j
8033020 d.log
5626152 d.ext
7214296 k
```

the goal is to organize the information and determine the filesystem structure, as presented below.

```
- / (dir)
  - a (dir)
    - e (dir)
      - i (file, size=584)
    - f (file, size=29116)
    - g (file, size=2557)
    - h.lst (file, size=62596)
  - b.txt (file, size=14848514)
  - c.dat (file, size=8504156)
  - d (dir)
    - j (file, size=4060174)
    - d.log (file, size=8033020)
    - d.ext (file, size=5626152)
    - k (file, size=7214296)
```

After doing that, we should be able to answer the part 1 question: "Find all of the directories with a total size of at most 100000. What is the sum of the total sizes of those directories?".

### Design Phase

The problem breakdown follows:

1. Parse the input: differentiate commands execution and commands output
2. Identify the "current working directory" and sum all the file sizes. Being paths, we should keep track of the absolute path every time we work on a folder, since it may be possible to have folders with the same name in different absolute positions.
3. Create the mapping `absolute path:size` for every directory.
4. Filter the mapped directories, searching for all the directories that satisfy the condition and summing up the sizes for getting the part 1 solution

### Input parsing: tf.data.Dataset.scan

[tf.data.Dataset.scan](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#scan) is a transformation that scans a function across an input dataset. It's the stateful counterpart of [tf.data.Dataset.map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map). The state is carried on on every iteration through the transformation `scan_func(old_state_input_element) -> (new_state, output_element)` via the state tensors.

The idea is to use the `scan` method for:

1. Parse the lines. Understand when the line is a command execution or the command output.
2. Whenever we change directory (command `cd`), create a new state that contains the "full path" we'll be visiting on the next iteration, together with the initial size (0). Thus, the initial state for every new folder will be `path 0`. We use a string so we can split the state, and extract the path and the size.
3. Every time we encounter something that's not a command (thus, is a command output) we should understand if it's a file (because a directory has no size). If it's a file, parse its size and carry on the state for this path, summing the filesize (e.g. at step i the state is `/this/path 10`. Then we encounter a file with size 100 at step i+1, thus the state now becomes `/this/path 110`).
4. The `scan` method produces an output value on every iteration, thus we can produce the state as the output value, but only when we change state (so we computed the complete size of a folder). By default, thus, we can use the empty string as output (so later on we can filter the output after applying the `scan` function) and produce an output in the format `path size` only when the path has been analyzed
5. The input contains different `cd` invocations. We can change directory using absolute or relative paths. Thus, we need to take care of this. Doing it with TensorFlow inside the `scan` method hasn't been possible - at least, I haven't found an easy solution. So, for now, we'll build an absolute path every time without considering the changes to relative locations. E.g. `cd /a/b/c` followed by `cd ..` will become `/a/b/c/../` instead of `/a/b/`. Both paths represent the same absolute location, thus we'll take care of this later on.


To correctly implement the algorithm described above, we need to add a "fake line" to the dataset, with a `cd` command, since we trigger the output of the accumulated state when we change directory. Moreover, we need to define a meaningful initial state. We can assume that the input always starts from the root directory `/`.

```python
dataset = dataset.concatenate(tf.data.Dataset.from_tensors(tf.constant("$ cd /")))
initial_state = tf.constant("/ 0")
```

Here's the implementation of the `scan_func` function that we'll use the loop over the dataset and extract the pairs `/x/y/z/../../i size`.

Take your time to read the implementation, it's not trivial.


```python
def func(old_state, line):
    is_command = tf.strings.regex_full_match(line, r"^\$.*")
    new_state = old_state
    if is_command:
        if tf.strings.regex_full_match(line, r"\$ cd .*"):
            dest = tf.strings.split([line], " ")[0][-1]
            if tf.equal(dest, "/"):
                new_state = tf.constant("/ 0")
            else:
                old_path = tf.strings.split([old_state], " ")[0][0]
                new_state = tf.strings.join(
                    [tf.strings.join([old_path, dest], "/"), "0"], " "
                )
    else:
        split = tf.strings.split([line], " ")[0]
        if tf.not_equal(split[0], "dir"):
            size = tf.strings.to_number(split[0], tf.int64)
            state_size = tf.strings.split([old_state], " ")[0]
            if tf.equal(tf.shape(state_size, tf.int64)[0], 1):
                old_size = tf.constant(0, tf.int64)
            else:
                old_size = tf.strings.to_number(state_size[1], tf.int64)

            partial_size = size + old_size
            new_state = tf.strings.join(
                [
                    tf.strings.split(old_state, " ")[0],
                    tf.strings.as_string(partial_size),
                ],
                " ",
            )

    if tf.not_equal(new_state, old_state):
        output_value = new_state
    else:
        output_value = tf.constant("")

    return new_state, output_value

intermediate_dataset = dataset.scan(initial_state, func)
````

[Once again](/tensorflow/2022/12/21/advent-of-code-tensorflow-day-5/#parsing-strings-tfstrings--regex) `tf.strings` and regex are the tools used for parsing the input and understand (thanks to the regular expression) whether we are parsing a command or not.

Iterating over `intermediate_dataset` we can get a list of elements such as

```
['', '/ 14848514', '/ 23352670', ...., '//a/e/../../d 17719346', '//a/e/../../d 24933642', '/ 0']
```

This scan-based solution created a new state every time a directory has been changed. However, the output produced while looping over the scan-generated dataset is the partial value computed while traversing a directory. For example, the first value of the list is `14848514` that's precisely `b.txt (file, size=14848514)`. The second value for `/` is `23352670` that's `b.txt + c.dat`.

Thus, we'll need to filter these values later on, after having resolved the absolute paths (e.g. going from `//a/e/../../d` to `/d`).

### Resolving paths: not pure TensorFlow solution

Removing empty lines from the `intermediate_dataset` is trivial and can be done by using the `tf.data.Dataset.filter` method.

```python
filtered_dataset = intermediate_dataset.filter(
    lambda line: tf.strings.regex_full_match(line, "^.* \d*$")
).map(lambda line: tf.strings.regex_replace(line, r"\/\/", "/"))
```

In this way, we just filtered all the empty lines and kept only the lines ending with a number. Moreover, we also replaced all the double slashes with a single slash, using a simple regex.

However, the very tough problem to solve is designing a pure TensorFlow solution for resolving all the relative paths. I haven't been able to design such a solution, and thus I had to use the Python `pathlib` module, unfortunately. It looks like cheating since this is not a pure TensorFlow solution anymore :(

```python

def gen(ds):
    def resolve():
        for pair in ds:
            path, count = tf.strings.split([pair], " ")[0]
            path = Path(path.numpy().decode("utf-8")).resolve().as_posix()
            yield path, count.numpy().decode("utf-8")

    return resolve

filtered_dataset = tf.data.Dataset.from_generator(
    gen(filtered_dataset), tf.string, output_shapes=[2]
)
```

Apart from this small cheat, so far so good. The `filtered_dataset` now contains absolute paths (repeated) and the value computed while traversing that path. So, as anticipated, we need to filter these values computed while traversing the path. They are sequential, so we can once again use the `tf.data.Dataset.scan` method to change-state every time the path changes, and produce only valid values.

```python
def mapper(old_state, pair):
    old_path = old_state[0]
    new_path = pair[0]
    output_value = tf.constant(["", ""])
    if tf.logical_or(
        tf.equal(old_path, "fake_path"), tf.equal(new_path, "fake_path")
    ):
        output_value = tf.constant(["", ""])
    elif tf.not_equal(old_path, new_path):
        output_value = old_state

    return pair, output_value

initial_state = tf.constant(["fake_path", "-1"])
filtered_dataset = (
    filtered_dataset.concatenate(tf.data.Dataset.from_tensors(initial_state))
    .scan(initial_state, mapper)
    .filter(
        lambda pair: tf.logical_and(
            tf.greater(tf.strings.length(pair[0]), 0), tf.not_equal(pair[1], "0")
        )
    )
)
```

`filtered_dataset` will now produce only the pairs containing the full path of each folder and its size, **without** considering the subfolders.

- `('/', '23352670')`
- `('/a', '94269')`
- `('/a/e', '584')`
- `('/d', '24933642')`

We now need to map the full paths with their corresponding full-size. In this way, we can easily answer the puzzle question.

### MutableHashTable: mapping paths to their size

We can, [once again](/tensorflow/2022/12/21/advent-of-code-tensorflow-day-5/#counting-crates-mutable-hashmaps), use a Mutable hashmap to iteratively create the correspondence between the full path and its size, also considering the sub-folders.

The algorithm is straightforward. We loop over the `filtered_dataset` and verify if the path is present. If it's not present, we insert the key-value pair `path,size` in the mutable hashmap. Otherwise, we split the path using the directory separator (`/`) and iteratively add the value to all the parent paths already inserted in the hashmap.


```python
lut = tf.lookup.experimental.MutableHashTable(tf.string, tf.int64, default_value=0)
for pair in filtered_dataset:
    path, value = pair[0], tf.strings.to_number(pair[1], tf.int64)
    parts = tf.strings.split(path, "/")

    if tf.logical_and(tf.equal(parts[0], parts[1]), tf.equal(parts[0], "")):
        keys = ["/"]
        old = lut.lookup(keys)[0]
        new = old + value
        lut.insert(keys, [new])
    else:
        for idx, part in enumerate(parts):
            if tf.equal(part, ""):
                keys = ["/"]
            else:
                l = [tf.constant("")] + parts[1 : idx + 1]
                j = tf.strings.join(l, "/")
                keys = [j]
            old = lut.lookup(keys)[0]
            new = old + value
            lut.insert(keys, [new])
```

The `lut` variable now contains all the information we need to solve the problem. The keys of the hashmap are all the available paths, the values are the full-size (considering subfolders) of every path.

Thus, we can answer the puzzle question in a few lines:

```python
paths, sizes = lut.export()
print(paths, sizes)
tf.print(
    "part 1: ",
    tf.reduce_sum(tf.gather(sizes, tf.where(tf.math.less_equal(sizes, 100000)))),
)
```

Problem solved! Let's go straight to part 2.

## [Day 7: No Space Left On Device: part 2](https://adventofcode.com/2022/day/7)

Luckily enough, part 2 presents us with an easy-to-solve problem. The puzzle tells us that our total disk space available is 70000000, and we need to install an update whose size is 30000000, but we don't have enough space for installing it!
The puzzle tells us to find the smallest directory that, if deleted, would free up enough space on the filesystem and print the total size of that directory.

In our lookup table (the mutable hashtable) we have everything that's needed to solve the problem. We need to:

1. Find the required space as the difference between the update size and the free space already present in the filesystem
2. Find all the folders whose size is "big enough" to contain the update (that means, to be deleted to install the update).
3. Get the smallest folder among the ones that satisfy the above criteria.

The code is a 1:1 mapping with these points.

```python
update_size = 30000000
free_space = 70000000 - lut.lookup("/")
required_space = update_size - free_space

big_enough = tf.gather(
    sizes, tf.where(tf.math.greater_equal(sizes - required_space, 0))
)
tf.print("part 2: ", tf.gather(big_enough, tf.math.argmin(big_enough, axis=0)))
```

Here we go! Day's 7 problem solved!

## Conclusion

You can see the complete solution in folder `7` in the dedicated GitHub repository (in the `2022` folder): [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

Solving problem 7 demonstrated all the weaknesses of the framework while working with strings and with paths in particular. There are no utilities for working with paths in graph mode, and thus this solution ended-up to use a `pathlib` utility for resolving the relative paths to their full paths counterparts. In general, `tf.data.Dataset.scan` has demonstrated to be a powerful method for keeping track of a state while iterating over a dataset, although it's not perfectly natural to solve problems like that in this way. Nonetheless, it's a fun exercise.


If you missed the article about the previous days’ solutions, here's a handy list

- [Advent of Code 2022 in pure TensorFlow - Days 1 & 2](/tensorflow/2022/12/04/advent-of-code-tensorflow-day-1-and-2/).
- [Advent of Code 2022 in pure TensorFlow - Days 3 & 4](/tensorflow/2022/12/11/advent-of-code-tensorflow-day-3-and-4/).
- [Advent of Code 2022 in pure TensorFlow - Day 5](/tensorflow/2022/12/21/advent-of-code-tensorflow-day-5/)
- [Advent of Code 2022 in pure TensorFlow - Day 6](/tensorflow/2022/12/27/advent-of-code-tensorflow-day-6/)

For any feedback or comment, please use the Disqus form below - thanks!
