---
layout: post
title: "Advent of Code 2021 in pure TensorFlow - day 8"
date: 2021-12-28 08:00:00
categories: tensorflow
summary: "The day 8 challenge is, so far, the most boring challenge faced 😅. Designing a TensorFlow program - hence reasoning in graph mode - would have been too complicated since the solution requires lots of conditional branches. A known AutoGraph limitation forbids variables to be defined in only one branch of a TensorFlow conditional if the variable is used afterward. That's why the solution is in pure TensorFlow eager."
authors:
    - pgaleone
---

The day 8 challenge is, so far, the most boring challenge faced 😅. Designing a TensorFlow program - hence reasoning in graph mode - would have been too complicated since the solution requires lots of conditional branches. A [known AutoGraph limitation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#:~:text=AutoGraph%20forbids%20variables%20to%20be%20defined%20in%20only%20one%20branch%20of%20a%20TensorFlow%20conditional%2C%20if%20the%20variable%20is%20used%20afterwards) forbids variables to be defined in only one branch of a TensorFlow conditional if the variable is used afterward. That's why the solution is in pure TensorFlow eager.

## [Day 8: Seven Segment Search](https://adventofcode.com/2021/day/8)

You can click on the title above to read the full text of the puzzle. The TLDR version is:

The input dataset is something like this

```
be cfbegad cbdgef fgaecd cgeb fdcge agebfd fecdb fabcd edb | fdgacbe cefdb cefbgd gcbe
edbfga begcd cbg gc gcadebf fbgde acbgfd abcde gfcbed gfec | fcgedb cgb dgebacf gc
```

Where every character (from `a` to `g`) represents a segment of a [seven-segment display](https://en.wikipedia.org/wiki/Seven-segment_display) turned on.

On the **left** of the separator character (`|`), we find some "visualization attempts", on the **right**, instead, we find the **display output**. Our goal, of part 2, is to decode the 4 digits of the display output.

Unfortunately, the mapping from the segments switches to the display output is broken. Let's say the correct mapping is the one that follows

```
 aaaa
b    c
b    c
 dddd
e    f
e    f
 gggg
```

Segment `a` turned on, means to turn on all the four `a` on top, segment `g` turned on means to turn on all the 4 `g` at the bottom, and so on.

As you can guess from the last digit of the second line of the example dataset (`gc`), that line is not a valid digit, in fact, if rendered it looks like this:

```

     c
     c


 gggg
```

Note: every line is independent - the segments `g` on one line, can turn on a different segment on another line.

The challenge presented in part one gives us some hints on how to start decoding the output digits.

### Design phase: part one

The first part of the puzzle requires us to count how many 1,4,7, and 8 are in the output digits.
These digits are particular since these are the only ones that require to turn on a unique number of segments:

- One: two segments
- Four: four segments
- Seven: three segments
- Eight: seven segments

All the other numbers are ambiguous and require different decoding - we'll see it in part two.

### Input pipeline

We create a `tf.data.Dataset` object for reading the text file line-by-line [as usual](/tensorflow/2021/12/11/advent-of-code-tensorflow/#input-pipeline). We can separately consider the "visualization attempts" (also called signal patterns) and the output digits. For solving the first part of the puzzle we only need to focus on the latter.

```python
dataset = (
    tf.data.TextLineDataset("input")
    .map(lambda line: tf.strings.split(line, " | "))
    .map(lambda lines: tf.strings.split(lines, " "))
)
```

The dataset object is an iterable object that produces pairs of `tf.string`. The first element contains a tensor with 10 strings (the signal patterns) the second element contains a batch with 4 strings (the output digits).

### Decoding and counting

Part one is trivial, we just need a `tf.Variable` to count the number of 1,4,7, and 8 found in the output digits, and a way to detect them.

The detection is straightforward since we just need to get the lengths of the strings and check if the length (number of turned-on segments) matches the expected one.

```python
count = tf.Variable(0, trainable=False, dtype=tf.int64)
for _, output_digits in dataset:
    lengths = tf.strings.length(output_digits)
    one = tf.gather_nd(lengths, tf.where(tf.equal(lengths, 2)))
    four = tf.gather_nd(lengths, tf.where(tf.equal(lengths, 4)))
    seven = tf.gather_nd(lengths, tf.where(tf.equal(lengths, 3)))
    eight = tf.gather_nd(lengths, tf.where(tf.equal(lengths, 7)))
    count.assign_add(
        tf.cast(
            tf.reduce_sum(
                [tf.size(one), tf.size(four), tf.size(seven), tf.size(eight)]
            ),
            tf.int64,
        )
    )
tf.print("Part one: ", count)
```

Here we go! Part one has already been completed!

## Design phase: part 2

Part two requires fully decoding the output digits and summing them all. For example, let's way we decoded the output digits presented at the beginning at:

- `fdgacbe cefdb cefbgd gcbe`: 8394
- `fcgedb cgb dgebacf gc`: 9781

The puzzle goal is to find the sum of all the output digits, in this example: 8394 + 9781 = 18175.

How can we decode these digits? We need to use our knowledge on how to decode the trivial digits (1,4,7,8) and use the visualization attempts to extract some meaningful information.

For example, let's say we already decoded from the output - or from the visualization attempts - the segments of the digit 4 and we are trying to decode a pattern that contains `5` segments turned on.

The digits that can be rendered with 5 segments are 2, 3, and 5.

The four we already decoded is rendered as follows:

```
 ....
b    c
b    c
 dddd
.    f
.    f
 ....
 ```

We need to compare the 2, 3, and 5 segments with the segments decoded for the 4.

- A 5 has 3 segments in common with 4.
- A 3 has 3 segments in common with 4.
- A 2 has 2 segments in common with 4. Found!

We can, thus, assert that if our input of length 5 has 2 segments in common with the pattern we are trying to decode, we have found the pattern of the number 2.

Finding the common characters is trivial using the TensorFlow sets support.

### TensorFlow set operations

The TensorFlow [`tf.sets`](https://www.tensorflow.org/api_docs/python/tf/sets) module offers the basic functionalities needed for working with sets. It's possible to compute the difference, the intersection, the union, and compute the number of unique elements in a set.

The sets are just `tf.Tensor` with the elements placed in the last dimension.

Knowing that we are ready to decode our digits.

## Complete decoding

First, we can define a helper function for easily decoding the easy digits (the one univocally identified by the number of segments) and returning the characters decoded.

```python
def search_by_segments(digits_set, num_segments):
    lengths = tf.strings.length(digits_set)
    where = tf.where(tf.equal(lengths, num_segments))
    number = tf.gather_nd(lengths, where)
    if tf.greater(num_found, 0):
        segments = tf.gather_nd(digits_set, where)[0]
    else:
        segments = tf.constant("")
    # segments (a,b,c)
    return tf.strings.bytes_split(segments)
```

We can loop over every entry of the dataset - that must be independently be decoded - and first, try to decode all the trivial digits (in the output or in the patterns).

For being solvable, every dataset entry must be at least one trivial digit, otherwise, the problem won't have a solution.

```python
count.assign(0)  # use count for sum

for signal_patterns, output_digits in dataset:
    # reverse because we compute from units to decimals, ...
    output_digits = tf.reverse(output_digits, axis=[0])
    all_digits = tf.concat([output_digits, signal_patterns], axis=0)
    lengths = tf.strings.length(all_digits)

    eight_chars = search_by_segments(all_digits, 7)
    four_chars = search_by_segments(all_digits, 4)
    seven_chars = search_by_segments(all_digits, 3)
    one_chars = search_by_segments(all_digits, 2)
    zero_chars = [""]
    two_chars = [""]
    three_chars = [""]
    five_chars = [""]
    six_chars = [""]
    nine_chars = [""]
```

Then, we can start searching for all the ambiguous patterns (e.g. the digits with 5 segments) and try to decode them with the info we have (`eight_chars`, `four_chars`, `seven_chars`, `one_chars`).

```python
# All the 5 segments: 2, 3, 5
five_segments = tf.strings.bytes_split(
    tf.gather_nd(all_digits, tf.where(tf.equal(lengths, 5)))
)
if tf.greater(tf.size(five_segments), 0):
    for candidate in five_segments:
        candidate_inter_seven = tf.sets.intersection(
            tf.expand_dims(candidate, axis=0),
            tf.expand_dims(seven_chars, axis=0),
        )
        candidate_inter_four = tf.sets.intersection(
            tf.expand_dims(candidate, axis=0),
            tf.expand_dims(four_chars, axis=0),
        )
        candidate_inter_one = tf.sets.intersection(
            tf.expand_dims(candidate, axis=0),
            tf.expand_dims(one_chars, axis=0),
        )
        # Use 7 as a reference:

        # A 2 has 2 in common with 7. I cannot identify it only with this
        # because also 2 has 2 in common with 7.

        # A 3 has 3 in common with 7. I can identify the 3 since 2 and 5 have only 2 in common.

        # 5 has 2 in common with 7. Cannot identify because of the 2.

        # Hence for identify a 2/5 I need a 7 and something else.
        # If I have a four:
        # A 2 has 2 in common with 7 and 2 in common with 4. Found!
        # A 5 has 2 in common with 7 and 3 in common with 4. Found!

        # If I have a one
        # A 2 has 2 in common with 7 and 1 in common with 1. Cannot identify.
        # A 5 has 2 in common with 7 and 1 in common with 1. Cannot identify.
        if tf.greater(tf.size(seven_chars), 0):
            if tf.equal(tf.size(candidate_inter_seven), 3):
                three_chars = candidate
            elif tf.logical_and(
                tf.greater(tf.size(four_chars), 0),
                tf.equal(tf.size(candidate_inter_seven), 2),
            ):
                if tf.equal(tf.size(candidate_inter_four), 2):
                    two_chars = candidate
                elif tf.equal(tf.size(candidate_inter_four), 3):
                    five_chars = candidate

        # use 4 as a reference

        # A 2 has 2 in common with 4. Found!
        # A 5 has 3 in common with 4.
        # A 3 has 3 in common with 4.

        # To find a 5,3 i need something else. Useless to check for seven, already done.
        # A 5 has 3 in common with 4 and 1 in common with 1. Found!
        # A 3 has 3 in common with 2 and 2 in common with 1. Found!

        if tf.greater(tf.size(four_chars), 0):
            if tf.equal(tf.size(candidate_inter_four), 2):
                two_chars = candidate
            if tf.logical_and(
                tf.equal(tf.size(candidate_inter_four), 3),
                tf.greater(tf.size(one_chars), 0),
            ):
                if tf.equal(tf.size(candidate_inter_one), 1):
                    five_chars = candidate
                else:
                    three_chars = candidate
```

The code is commented and explains how to reason. The very same reasoning must be applied to the other ambiguous digits with 6 segments (6,9,0), but it won't be presented in the article since it's identical to the code presented above. You can read the complete code [here](https://github.com/galeone/tf-aoc/blob/main/8/main.py).

In the end, we should decode the `output_digits` with our accrued knowledge. In practice, we'll be able to decode an output digit if the intersection between the output digit characters and the decoded digit character is empty.

```python
for position, digit in enumerate(output_digits):
    digit = tf.strings.bytes_split(digit)
    for num, k in enumerate(
        [
            zero_chars,
            one_chars,
            two_chars,
            three_chars,
            four_chars,
            five_chars,
            six_chars,
            seven_chars,
            eight_chars,
            nine_chars,
        ]
    ):
        difference_1 = tf.sets.difference(
            tf.expand_dims(digit, axis=0), tf.expand_dims(k, axis=0)
        )
        difference_2 = tf.sets.difference(
            tf.expand_dims(k, axis=0), tf.expand_dims(digit, axis=0)
        )
        if tf.logical_and(
            tf.equal(tf.size(difference_1), 0),
            tf.equal(tf.size(difference_2), 0),
        ):
            count.assign_add(num * 10 ** position)
```

Problem 8 completed!

## Conclusion

You can see the complete solution in folder `8` on the dedicated Github repository: [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

The challenge in the challenge of using only TensorFlow for solving the problem is slowly progressing, so far I solved all the puzzles up to Day 12 (inclusive). So get ready for at least 4 more articles :) Let's see when (and if!) TensorFlow alone won't be enough.

If you missed the articles about the previous days' solutions, here's a handy list:

- [Day 1](/tensorflow/2021/12/11/advent-of-code-tensorflow/)
- [Day 2](/tensorflow/2021/12/12/advent-of-code-tensorflow-day-2/)
- [Day 3](/tensorflow/2021/12/14/advent-of-code-tensorflow-day-3/)
- [Day 4](/tensorflow/2021/12/17/advent-of-code-tensorflow-day-4/)
- [Day 5](/tensorflow/2021/12/22/advent-of-code-tensorflow-day-5/)
- [Day 6](/tensorflow/2021/12/25/advent-of-code-tensorflow-day-6/)
- [Day 7](/tensorflow/2021/12/28/advent-of-code-tensorflow-day-7/)

The next article will be about my solution to Day 9 problem. This solution is really interesting IMO because I solved it using lots of computer vision concepts like image gradients and flood fill algorithm. Maybe an unconventional - but working - approach.

For any feedback or comment, please use the Disqus form below - thanks!
