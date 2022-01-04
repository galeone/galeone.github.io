---
layout: post
title: "Advent of Code 2021 in pure TensorFlow - day 10"
date: 2022-01-04 08:00:00
categories: tensorflow
summary: "The day 10 challenge projects us in the world of syntax checkers and autocomplete tools. In this article, we'll see how TensorFlow can be used as a generic programming language for implementing a toy syntax checker and autocomplete."
authors:
    - pgaleone
---

The day 10 challenge projects us in the world of syntax checkers and autocomplete tools. In this article, we'll see how TensorFlow can be used as a generic programming language for implementing a toy syntax checker and autocomplete.


## [Day 10: Syntax Scoring ](https://adventofcode.com/2021/day/10)

You can click on the title above to read the full text of the puzzle. The TLDR version is:

The puzzle gives us a dataset of lines with different lengths, each of them contains several **chunks**. A chunk is nothing but some text that starts with a character and closes with another character. In particular:

- If a chunk opens with `(`, it must close with `)`.
- If a chunk opens with `[`, it must close with `]`.
- If a chunk opens with `{`, it must close with `}`.
- If a chunk opens with `<`, it must close with `>`.

The dataset looks like

```
[(()[<>])]({[<{<<[]>>(
{([(<{}[<>[]}>{[]{[(<()>
(((({<>}<{<{<>}{[]{[]{}
[[<[([]))<([[{}[[()]]]
[{[{({}]{}}([{[{{ '{{' }}{}}([]
{<[[]]>}<{[{[{[]{()[[[]
[<(<(<(<{}))><([]([]()
<{([([[(<>()){}]>(<<{{ '{{' }}
<{([{{ '{{' }}}}[<[[[<>{}]]]>[]]
```

where some lines are **corrupted** others are **incomplete**.

- A **corrupted** line is one where a chunk closes with the wrong character.
- An **incomplete** line is missing some closing characters at the end of the line.

Part one asks us to implement a **syntax checker**. So, given the example dataset, our program should detect errors like

> Expected `]`, but found `}` instead.

at the third line (that's the first corrupted line): `{([(<{}[<>[]}>{[]{[(<()>`.

Every time we detect a corrupted line, we should stop our parsing for that line, **take the first illegal character** on the line and use this **lookup table** to compute a score.

| Char | Points |
| --- | --- |
|`)`|3|
|`]`|57|
|`}`|1197|
|`>`|25137|

The final score is the sum of all the errors detected in all the corrupted lines.

### Design phase: part one

For detecting the first wrong character we need to [tokenize](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization) the string and consider every character has a language token.

We know from the rules that every opening token as a corresponding closing token. This means that every time we found an opening token we can **push into a stack** the corresponding closing token.
As soon as we detect a closing token, we pop the expected one from the stack. If the popped one does not correspond to the token under analysis: we found a **corrupted** line, and we also know what the expected char is, and what we found instead. Problem solved!

If instead, we reached the end of the line and our stack is not empty, it means or line is **incomplete**.

### Input pipeline

We create a `tf.data.Dataset` object for reading the text file line-by-line [as usual](/tensorflow/2021/12/11/advent-of-code-tensorflow/#input-pipeline). Since we work with characters, we can directly map the function `tf.strings.bytes_split` that given a `tf.string` explodes it in a batch of `tf.string` with length 1.

```python
dataset = tf.data.TextLineDataset("input").map(tf.strings.bytes_split)
```

We can now define our TensorFlow program named `Tokenizer`.

### Finding corrupted lines

We can define a `Tokenizer` class that contains all the mapping defined in the requirements. We have all the opening tokens, the closing tokens, the mapping between opening and closing, and the scores to use when we find a wrong closing token.

```python
class Tokenizer(tf.Module):
    def __init__(self):
        super().__init__()

        self._opening_tokens = tf.constant(["(", "[", "{", "<"])
        self._closing_tokens = tf.constant([")", "]", "}", ">"])

        self._syntax_score_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                self._closing_tokens,
                tf.constant([3, 57, 1197, 25137], tf.int64),
            ),
            default_value=tf.constant(-1, tf.int64),
        )

        self._open_close = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                self._opening_tokens,
                self._closing_tokens,
            ),
            default_value="",
        )

        self._close_open = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                self._closing_tokens,
                self._opening_tokens,
            ),
            default_value="",
        )

        self._pos = tf.Variable(0, dtype=tf.int64)
        self._corrupted_score = tf.Variable(0, dtype=tf.int64)
```

TensorFlow offers us a constant hashtable that perfectly fits our needs: `tf.lookup.StaticHashTable`. All the data is constant and we know this mapping in advance.

Since we are writing a TensorFlow program, the `tf.Variable` objects must be defined outside the `@tf.function`-decorated methods. In this case, we defined the `_corrupted_score` that will hold the final score and the `_pos` score that will use to index a `tf.TensorArray` we use a **stack**.

The detection of the corrupted lines and the computation of the score is precisely what we described in the previous design phase.


```python
@tf.function
def corrupted(self, dataset):
    for line in dataset:
        stack = tf.TensorArray(tf.string, size=0, dynamic_size=True)
        self._pos.assign(0)
        for position in tf.range(tf.size(line)):
            current_token = line[position]
            if tf.reduce_any(tf.equal(current_token, self._opening_tokens)):
                stack = stack.write(tf.cast(self._pos, tf.int32), current_token)
                self._pos.assign_add(1)
            else:
                expected_token = self._open_close.lookup(
                    stack.read(tf.cast(self._pos - 1, tf.int32))
                )
                self._pos.assign_sub(1)
                if tf.not_equal(current_token, expected_token):
                    tf.print(
                        position,
                        ": expected: ",
                        expected_token,
                        " but found ",
                        current_token,
                        " instead",
                    )
                    self._corrupted_score.assign_add(
                        self._syntax_score_table.lookup(current_token)
                    )
                    break
    return self._corrupted_score
```

Every line parsing requires its own stack, that's why the `tf.TensorArray` stack is defined inside the loop.

### Execution

Here we go!

```python
tokenier = Tokenizer()

tf.print("Part one: ", tokenier.corrupted(dataset))
```

Part one is easily solved. Let's see what part two is about.

## Design phase: part 2

This second part asks us to discard the corrupted lines and focus on the **incomplete** lines. The puzzle wants us to implement an autocomplete system for incomplete (but not corrupted) lines.

For example, given the line `[({(<(())[]>[[{[]{<()<>>` our autocomplete system should be able to generate the correct sequence of closing characters: `}}]])})]`.

The puzzle goal is to compute a score following this rule:

Start with a total score of 0. Then, for each character, multiply the total score by 5 and then increase the total score by the point value given for the character in the following table:

| Char | Points |
| --- | --- |
|`)`| 1 |
|`]`| 2 |
|`}`| 3 |
|`>`| 4 |

The puzzle asks us to find "the winner". The winner is found by sorting all of the scores and then taking the middle score.

### Implementing the autocomplete

We know from the previous design phase how to detect **incomplete** lines (when the stack is not empty) and inside the stack, we have **in reversed order** the expected closing characters. Implementing the required algorithm is straightforward.

We first need to add a new lookup table that associates the points with the closing tokens

```python
self._autocomplete_score_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        self._closing_tokens,
        tf.constant([1, 2, 3, 4], tf.int64),
    ),
    default_value=tf.constant(-1, tf.int64),
)
```

then, we can implement the `incomplete` method by replicating the very same code used for the `corrupted` method and extending it a bit. Since we need to find "the winner" we need another `tf.TensorArray` for storing every autocomplete score computed.

```python
@tf.function
def incomplete(self, dataset):
    scores = tf.TensorArray(tf.int64, size=0, dynamic_size=True)
    for line in dataset:
        # identical loop code of `corrupted`
        # [ omitted ]
        # visible here
        # https://github.com/galeone/tf-aoc/blob/main/10/main.py#L87

        if tf.not_equal(self._pos, 0):  # stack not completely unrolled
            unstacked = tf.squeeze(
                tf.reverse(
                    tf.expand_dims(stack.stack()[: self._pos], axis=0), axis=[1]
                )
            )
            closing = self._open_close.lookup(unstacked)
            tf.print("Unstacked missing part: ", closing, summarize=-1)

            # Use pos variable as line score
            self._pos.assign(0)
            for idx in tf.range(tf.shape(closing)[0]):
                char = closing[idx]
                self._pos.assign(self._pos * 5)
                self._pos.assign_add(self._autocomplete_score_table.lookup(char))

            scores = scores.write(scores.size(), self._pos)

    # sort the scores
    scores_tensors = tf.sort(scores.stack())
    # tf.print(scores_tensors)
    return scores_tensors[(tf.shape(scores_tensors)[0] - 1) // 2]
```

Here we go! Challenge 10 is solved in pure TensorFlow just using a pair of stacks and some static lookup table :)

## Conclusion

You can see the complete solution in folder `10` on the dedicated Github repository: [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

Solving this problem has been straightforward and TensorFlow has proved enough flexibility for being used to solve all the problems faced so far, without the need of any external library. Will we find a problem impossible to solve in pure TensorFlow? Who knows! I'm currently solving problem 13 and TensorFlow is still showing to be flexible enough for solving it.

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

The next article will be about my solution to Day 11 problem. That problem shares some similarities with the [Day 9](/tensorflow/2022/01/01/advent-of-code-tensorflow-day-9/) solution - I'll re-use some computer vision concepts like the pixel neighborhood since the problem is again organized as a grid of numbers with some relations among each other.

For any feedback or comment, please use the Disqus form below - thanks!
