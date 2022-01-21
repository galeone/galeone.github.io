---
layout: post
title: "Wrap up of Advent of Code 2021 in pure TensorFlow"
date: 2022-01-21 08:00:00
categories: tensorflow
summary: "A wrap up of my solutions to the Advent of Code 2021 puzzles in pure TensorFlow"
authors:
    - pgaleone
---

This is the first time I participate in the [Advent of Code](https://adventofcode.com/) challenge and it has been really fun!

I decided to solve every puzzle in "pure TensorFlow" - that means, solving the puzzle without any other library, and trying to write "TensorFlow programs".

A TensorFlow program is a pure-TensorFlow object that **describes** the computation. This is a nice feature because it allows describing a solution, exporting the computation in a language-agnostic format (the [SavedModel](https://www.tensorflow.org/guide/saved_model)), and running the computation everywhere (what we only need is the [C TensorFlow runtime](https://www.tensorflow.org/install/lang_c)). TensorFlow programs are `tf.function`-decorated functions/objects.

I haven't solved all the puzzles because this has been a pastime for the holidays; holidays that are now gone. I solved precisely half of the puzzles, 12.5/25.

Day 13 problem has been only partially solved because there's something strange going on in my code I haven't had the time to look carefully (but the code is on the Github repo if someone wants to progress!).

In the twelve articles I wrote, I tried to explain how I designed the solutions, how I implemented them, and  - when needed - focus on some TensorFlow features not widely used.

For example, in these articles, you can find descriptions and usages of [tf.TensorArray](https://www.tensorflow.org/api_docs/python/tf/TensorArray?hl=en), [tf.queue](https://www.tensorflow.org/api_docs/python/tf/queue), correct usages of [@tf.function](https://www.tensorflow.org/api_docs/python/tf/function), [tf.sets](https://www.tensorflow.org/api_docs/python/tf/sets), and many other now widely used (or often incorrectly used) features.

This article is just a wrap-up of my solutions. If you want to jump straight to the code here's the repo with the 12.5 tasks completed and also the first part of day 13: [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

## Day 1: Sonar Sweep

Solving a coding puzzle with TensorFlow doesn't mean throwing fancy machine learning stuff (without any reason) to the problem for solving it. On the contrary, I want to demonstrate the flexibility - and the limitations - of the framework, showing that TensorFlow can be used to solve any kind of problem and that the produced solutions have tons of advantages with respect to the solutions developed using any other programming languages.

Article: [Advent of Code 2021 in pure TensorFlow - day 1](/tensorflow/2021/12/11/advent-of-code-tensorflow/)

## Day 2: Dive!

A Solution to the AoC day 2 puzzle in pure TensorFlow. How to use Enums in TensorFlow programs and the limitations of tf.Tensor used for type annotation.

Article: [Advent of Code 2021 in pure TensorFlow - day 2](/tensorflow/2021/12/12/advent-of-code-tensorflow-day-2/)

## Day 3: Binary Diagnostic

A Solution to the AoC day 3 puzzle in pure TensorFlow. This challenge allows us to explore the TensorArray data type and find their limitations when used inside a static-graph context. We'll also use a tf.function experimental (but very useful) feature for avoiding useless retraces and reusing the same graph with tensors of different shapes.

Article: [Advent of Code 2021 in pure TensorFlow - day 3](/tensorflow/2021/12/14/advent-of-code-tensorflow-day-3/)

## Day 4: Giant Squid

Using tensors for representing and manipulating data is very convenient. This representation allows changing shape, organizing, and applying generic transformations to the data. TensorFlow - by design - executes all the data manipulation in parallel whenever possible. The day 4 challenge is a nice showcase of how choosing the correct data representation can easily simplify a problem.

Article: [Advent of Code 2021 in pure TensorFlow - day 4](/tensorflow/2021/12/17/advent-of-code-tensorflow-day-4/)

## Day 5: Hydrothermal Venture

The day 5 challenge is easily solvable in pure TensorFlow thanks to its support for various distance functions and the power of the tf.math package. The problem only requires some basic math knowledge to be completely solved - and a little bit of computer vision experience doesn't hurt.

Article: [Advent of Code 2021 in pure TensorFlow - day 5](/tensorflow/2021/12/22/advent-of-code-tensorflow-day-5/)

## Day 6: Lanternfish

The day 6 challenge has been the first one that obliged me to completely redesign for part 2 the solution I developed for part 1. For this reason, in this article, we'll see two different approaches to the problem. The former will be computationally inefficient but will completely model the problem, hence it will be easy to understand. The latter, instead, will be completely different and it will focus on the puzzle goal instead of the complete modeling.

Article: [Advent of Code 2021 in pure TensorFlow - day 6](/tensorflow/2021/12/25/advent-of-code-tensorflow-day-6/)

## Day 7: The Treachery of Whales

The day 7 challenge is easily solvable with the help of the TensorFlow ragged tensors. In this article, we'll solve the puzzle while learning what ragged tensors are and how to use them.

Article: [Advent of Code 2021 in pure TensorFlow - day 7](/tensorflow/2021/12/28/advent-of-code-tensorflow-day-7/)

## Day 8: Seven Segment Search

The day 8 challenge is, so far, the most boring challenge faced 😅. Designing a TensorFlow program - hence reasoning in graph mode - would have been too complicated since the solution requires lots of conditional branches. A known AutoGraph limitation forbids variables to be defined in only one branch of a TensorFlow conditional if the variable is used afterward. That's why the solution is in pure TensorFlow eager.

Article: [Advent of Code 2021 in pure TensorFlow - day 8](/tensorflow/2021/12/28/advent-of-code-tensorflow-day-8/)

## Day 9: Smoke Basin

The day 9 challenge can be seen as a computer vision problem. TensorFlow contains some computer vision utilities that we'll use - like the image gradient - but it's not a complete framework for computer vision (like OpenCV). Anyway, the framework offers primitive data types like tf.TensorArray and tf.queue that we can use for implementing a flood-fill algorithm in pure TensorFlow and solve the problem.

Article: [Advent of Code 2021 in pure TensorFlow - day 9](/tensorflow/2022/01/01/advent-of-code-tensorflow-day-9/)

## Day 10: Syntax Scoring

The day 10 challenge projects us in the world of syntax checkers and autocomplete tools. In this article, we'll see how TensorFlow can be used as a generic programming language for implementing a toy syntax checker and autocomplete.

Article: [Advent of Code 2021 in pure TensorFlow - day 10](/tensorflow/2022/01/04/advent-of-code-tensorflow-day-10/)

## Day 11: Dumbo Octopus

The Day 11 problem has lots in common with Day 9. In fact, will re-use some computer vision concepts like the pixel neighborhood, and we'll be able to solve both parts in pure TensorFlow by using only a tf.queue as a support data structure.

Article: [Advent of Code 2021 in pure TensorFlow - day 11](/tensorflow/2022/01/08/advent-of-code-tensorflow-day-11/)

## Day 12: Passage Pathing

Day 12 problem projects us the world of graphs. TensorFlow can be used to work on graphs pretty easily since a graph can be represented as an adjacency matrix, and thus, we can have a tf.Tensor containing our graph. However, the "natural" way of exploring a graph is using recursion, and as we'll see in this article, this prevents us to solve the problem using a pure TensorFlow program, but we have to work only in eager mode.

Article: [Advent of Code 2021 in pure TensorFlow - day 12](/tensorflow/2022/01/15/advent-of-code-tensorflow-day-12/)

## Conclusion

It has been a funny experience and I hope these articles can shed a light on the real capabilities of TensorFlow, and how to correctly use it *as a programming language*.

For any feedback or comment, please use the Disqus form below - thanks!
