---
layout: post
title: "Advent of Code 2022 in pure TensorFlow - Day 11"
date: 2023-03-26 08:00:00
categories: tensorflow
summary: "In this article, we'll show how to solve problem 11 from the Advent of Code 2022 (AoC 2022) using TensorFlow. We'll first introduce the problem and then provide a detailed explanation of our TensorFlow solution. The problem at hand revolves around the interactions of multiple monkeys inspecting items, making decisions based on their worry levels, and following a set of rules."
authors:
    - pgaleone
---

In this article, we demonstrate how to solve problem 11 of the Advent of Code 2022 using pure TensorFlow. While TensorFlow is primarily known for its applications in deep learning and neural networks, it offers powerful and flexible tools for working with tensors and performing computations on them. The problem at hand revolves around the interactions of multiple monkeys inspecting items, making decisions based on their worry levels, and following a set of rules. By leveraging TensorFlow's features, such as `tf.TensorArray`, `tf.data.Dataset.scan`, and `tf.function`, we can implement an efficient and elegant solution to this challenging puzzle. We will delve into the code, analyze different sections, and explain the rationale behind the various techniques used. Furthermore, this article provides insights into how TensorFlow can be employed for solving complex problems beyond its traditional applications in machine learning.

## [Day 11: Monkey in the Middle](https://adventofcode.com/2022/day/11)

In the Advent of Code 2022 problem 11, you are tasked with predicting the behavior of monkeys who have stolen your items. The monkeys operate based on your worry level for each item. The input consists of a description of multiple monkeys, including their starting items (represented by worry levels), an operation to modify the worry level, a test to decide where to throw the item, and the destination monkeys for both true and false test outcomes. The monkeys take turns inspecting and throwing items in rounds.

The input dataset looks like

```text
Monkey 0:
  Starting items: 79, 98
  Operation: new = old * 19
  Test: divisible by 23
    If true: throw to monkey 2
    If false: throw to monkey 3

Monkey 1:
  Starting items: 54, 65, 75, 74
  Operation: new = old + 6
  Test: divisible by 19
    If true: throw to monkey 2
    If false: throw to monkey 0

[...]
```

Each monkey has several attributes: Starting items, Operation, Test, If true, and If false. In the example above, Monkey 0 starts with items that have worry levels of 79 and 98. The monkey modifies the worry level by multiplying it by 19. It then checks if the modified worry level is divisible by 23. If it is, Monkey 0 throws the item to Monkey 2; otherwise, it throws the item to Monkey 3.

The monkeys take turns inspecting and throwing items in rounds. In the first round, Monkey 0 inspects the first item with a worry level of 79. So a simulation of the first round for the Monkey 0 looks like

```
Monkey inspects an item with a worry level of 79.
Worry level is multiplied by 19 to 1501.
Monkey gets bored with item. Worry level is divided by 3 to 500.
Current worry level is not divisible by 23.
Item with worry level 500 is thrown to monkey 3.
```

The goal is to find the two most active monkeys after 20 rounds for part 1 and 20000 for part 2.

The level of monkey business is determined by multiplying the total number of times these two most active monkeys inspected items over the rounds.


### Parsing the input

To begin, we read the input data using TensorFlow's `tf.data.TextLineDataset` and concatenate it with an empty line. This way, we can detect when the dataset ends and reset the monkey state accordingly. We then create a dataset using the `scan` method to extract information about the monkeys and their operations. The `init` function is used to process all the input dataset and forwarding a state on each iteration, being used inside the `scan` method. We use the `pos` variable to to keep track of the read line, knowing that in the dataset every monkey information is separated by an empty line from the next monkey info.

```python
dataset = tf.data.TextLineDataset(input_path.as_posix())
dataset = dataset.concatenate(tf.data.Dataset.from_tensors([""]))

monkey = tf.Variable(["", "", "", "", "", ""], dtype=tf.string)
monkey_id = tf.Variable(-1)
pos = tf.Variable(0)

initial_state = tf.constant(["", "", "", "", "", ""])

def init(old_state, line):

    if tf.equal(line, ""):
        monkey.assign(old_state, use_locking=True)
        pos.assign(0)
        return initial_state, True

    if tf.strings.regex_full_match(line, r"^Monkey \d*:$"):
        items = tf.strings.split(tf.strings.split([line], " ")[0][1], ":")[0]
        updates = [items]
    elif tf.equal(pos, 1):
        items = tf.strings.strip(tf.strings.split([line], ":")[0][1])
        updates = [items]
    elif tf.equal(pos, 2):
        op = tf.strings.strip(tf.strings.split([line], "="))[0][1]
        updates = [op]
    elif tf.equal(pos, 3):
        divisible_by = tf.strings.strip(tf.strings.split([line], " "))[0][-1]
        updates = [divisible_by]
    else:  # if tf.reduce_any([tf.equal(pos, 4), tf.equal(pos, 5)]):
        monkey_dest = tf.strings.strip(tf.strings.split([line], " "))[0][-1]
        updates = [monkey_dest]

    indices = tf.reshape(pos, (1, 1))
    new_state = tf.tensor_scatter_nd_update(old_state, indices, updates)
    pos.assign_add(1)

    return new_state, False

dataset = dataset.scan(initial_state, init)
```

### Applying Operations

We define the `apply_operation` function to perform the specified operation on the worry level according to the monkey's rules. This function takes the current worry level and the operation as inputs and returns the updated worry level after applying the operation.

```python
@tf.function
def apply_operation(worry_level, op):
    op = tf.strings.split([op], " ")[0]  # lhs, op, rhs
    ret = tf.constant(0, tf.int64)
    # lhs always = "old"
    if tf.strings.regex_full_match(op[2], r"^\d*$"):
        val = tf.strings.to_number(op[2], tf.int64)
    else:
        val = worry_level
    if tf.equal(op[1], "+"):
        ret = worry_level + val
    if tf.equal(op[1], "*"):
        ret = worry_level * val

    return ret
```

### Finding the Monkey Business

We create a function called `monkey_play` to simulate the monkeys' actions for a given number of rounds. Inside this function, we loop through the dataset and perform the required operations for each monkey based on their rules. We also keep track of the number of items inspected by each monkey in the `inspected_count` variable.

The problem is the very same for both parts, the first time it asks to simulate the monkey behavior a small number of times, while the second part asks us to simulate it for 10000. This makes the "trivial" implementation computationally impossible, and we need to use a mathematical trick to find a way to make this problem solvable.

We use a `tf.Variable` to switch between the part 1 and part 2 inside the `monkey_play` function.

```python
part = tf.Variable(1)

@tf.function
def monkey_play(rounds):
    items = tf.TensorArray(tf.int64, size=1, dynamic_size=True)
    operation = tf.TensorArray(tf.string, size=1, dynamic_size=True)
    divisible_test = tf.TensorArray(tf.int64, size=1, dynamic_size=True)
    throw_if_true = tf.TensorArray(tf.int32, size=1, dynamic_size=True)
    throw_if_false = tf.TensorArray(tf.int32, size=1, dynamic_size=True)

    for monkey_ready in dataset:
        if monkey_ready:
            idx = tf.strings.to_number(monkey[0], tf.int32)
            items = items.write(
                idx,
                tf.strings.to_number(tf.strings.split(monkey[1], ","), tf.int64),
            )
            operation = operation.write(idx, monkey[2])
            divisible_test = divisible_test.write(
                idx, tf.strings.to_number(monkey[3], tf.int64)
            )
            throw_if_true = throw_if_true.write(
                idx, tf.strings.to_number(monkey[4], tf.int32)
            )
            throw_if_false = throw_if_false.write(
                idx, tf.strings.to_number(monkey[5], tf.int32)
            )

    if tf.equal(part, 1):
        divisor = tf.constant(3, tf.int64)
    else:
        divisor = tf.reduce_prod(divisible_test.stack())

    for r in tf.range(rounds):
        # Now items contains all the starting items for every monkey
        # Let's play
        for m in tf.range(monkey_count):
            m_items = items.read(m)
            op = operation.read(m)
            test = divisible_test.read(m)

            for i in tf.range(tf.shape(m_items)[0]):
                worry_level = apply_operation(m_items[i], op)
                if tf.equal(part, 1):
                    worry_level //= divisor
                else:
                    worry_level = tf.math.mod(worry_level, divisor)

                if tf.equal(tf.math.mod(worry_level, test), 0):
                    dest = throw_if_true.read(m)
                else:
                    dest = throw_if_false.read(m)

                items = items.write(
                    dest,
                    tf.concat(
                        [items.read(dest), tf.expand_dims(worry_level, axis=0)],
                        axis=0,
                    ),
                )

                update = tf.tensor_scatter_nd_add(
                    inspected_count,
                    [[tf.cast(m, tf.int64)]],
                    [tf.constant(1, tf.int64)],
                )
                inspected_count.assign(update)

            items = items.write(m, [])
```

This function "simply" simulates the monkey business. Ignoring the separation between part 1 and part 2 (for now) let's only focus on the intensive use of `tf.TensorArray` done inside this function.

### Using tf.TensorArray

In our solution, we made use of `tf.TensorArray`, which is a mutable and dynamically-sized container for tensors. This data structure is useful when working with sequences of tensors that can change in size during runtime.

The reason we initialized the `tf.TensorArray` variables **inside** the `@tf.function`-decorated function instead of outside is related to the way TensorFlow traces and optimizes functions decorated with `@tf.function`. When a function is decorated with `@tf.function`, TensorFlow traces the computation defined by the function and generates a computation graph to optimize its execution. Initializing the `tf.TensorArray` variables outside the function would cause the traced computation to depend on external state, which can lead to unexpected behavior and difficulties in optimization.

By initializing the `tf.TensorArray` variables inside the decorated function, we ensure that the computation is self-contained, making it easier for TensorFlow to trace and optimize the computation. Additionally, this helps prevent potential issues that might arise due to external state changes during the lifetime of the TensorArray objects.

In summary, initializing `tf.TensorArray` variables inside the `@tf.function`-decorated function ensures a clean separation between the function's internal computation and external state, allowing TensorFlow to effectively trace, optimize, and execute the computation.

Let's go back to the solution...

```python
monkey_play(20)
top_values, _ = tf.math.top_k(inspected_count, k=2)
monkey_business = tf.reduce_prod(top_values)
tf.print("Part 1: ", monkey_business)
```

After running the `monkey_play` function for 20 rounds, we find the top 2 values of the `inspected_count` variable and calculate their product, which gives us the answer to part 1 of the problem.

### Part 2: Adapting for Larger Iterations

Honestly, I've spent some hour reasoning on how to make the second part to not computationally explode - I give up after a bit and I found a [spoiler](https://www.reddit.com/r/adventofcode/comments/zifqmh/comment/j0c0fcz/?utm_source=reddit&utm_medium=web2x&context=3) on reddit that clarified pretty well how to reason. I won't report the spoiler here, but it's worth reading to learn something new and well explained.

Anyway, the code of the `monkey_play` function already contains the implementation of the spoiler suggestion, therefore for us it's just a matter of chaning the variable `part` and run the code for 20000 iterations as requested. Of course, since we have `tf.Variable` declared (correctly) outside of the `@tf.function`-decorated functions, we need to reset the state before proceeding (the first line of the following snipped does it).

```python
inspected_count.assign(tf.zeros_like(inspected_count))
part.assign(2)
monkey_play(10000)
top_values, _ = tf.math.top_k(inspected_count, k=2)
monkey_business = tf.reduce_prod(top_values)
tf.print("Part 2: ", monkey_business)
```

## Conclusion

You can the solution in folder `11` in the dedicated GitHub repository (in the `2022` folder): [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

In this article, we demonstrated how to solve problem 11 of the Advent of Code 2022 using pure TensorFlow. By leveraging TensorFlow's powerful features and its ability to work with tensors, we were able to efficiently solve both parts of the problem. This unconventional approach showcases the versatility of TensorFlow beyond its typical use in machine learning and deep learning applications. By exploring different techniques and libraries, we can develop creative and efficient solutions to various computational problems.


If you missed the article about the previous days’ solutions, here's a handy list

- [Advent of Code 2022 in pure TensorFlow - Days 1 & 2](/tensorflow/2022/12/04/advent-of-code-tensorflow-day-1-and-2/).
- [Advent of Code 2022 in pure TensorFlow - Days 3 & 4](/tensorflow/2022/12/11/advent-of-code-tensorflow-day-3-and-4/).
- [Advent of Code 2022 in pure TensorFlow - Day 5](/tensorflow/2022/12/21/advent-of-code-tensorflow-day-5/)
- [Advent of Code 2022 in pure TensorFlow - Day 6](/tensorflow/2022/12/27/advent-of-code-tensorflow-day-6/)
- [Advent of Code 2022 in pure TensorFlow - Day 7](/tensorflow/2022/12/29/advent-of-code-tensorflow-day-7/)
- [Advent of Code 2022 in pure TensorFlow - Day 8](/tensorflow/2023/01/14/advent-of-code-tensorflow-day-8/)
- [Advent of Code 2022 in pure TensorFlow - Day 9](/tensorflow/2023/01/23/advent-of-code-tensorflow-day-9/)
- [Advent of Code 2022 in pure TensorFlow - Day 10](/tensorflow/2023/03/22/advent-of-code-tensorflow-day-10/)

For any feedback or comment, please use the Disqus form below - thanks!
