---
layout: post
title: "Advent of Code 2021 in pure TensorFlow - day 3"
date: 2021-12-14 08:00:00
categories: tensorflow
summary: "A Solution to the AoC day 3 puzzle in pure TensorFlow. This challenge allows us to explore the TensorArray data type and find their limitations when used inside a static-graph context. We'll also use a tf.function experimental (but very useful) feature for avoiding useless retraces and reusing the same graph with tensors of different shapes."
authors:
    - pgaleone
---

The day 3 challenge is very different from the easy challenges faced during [day 1](/tensorflow/2021/12/11/advent-of-code-tensorflow/) and [day 2](/tensorflow/2021/12/12/advent-of-code-tensorflow-day-2/). This time, we need to face a more difficult challenge and by doing so we'll explore some useful, although not widely used, TensorFlow features like `tf.TensorArray`.

Moreover, we'll find some limitations (bug?) of the `TensorArray` data type and we'll write some interesting utility that's not present in the TensorFlow standard library.

## [Day 3: Binary Diagnostic](https://adventofcode.com/2021/day/3): part one

You can click on the title above to read the full text of the puzzle. The TLDR version is:

You are given a dataset in the format

```
00100
11110
10110
10111
10101
01111
00111
11100
10000
11001
00010
01010
```

The text asks to generate two new **binary** numbers called *gamma rate* and *epsilon rate*. The goal is to multiply these numbers together and give the result in **decimal**. This result is called "power consumption".

- **Gamma rate**: the number can be determined by finding the **most common bit in the corresponding position** of all numbers. For example, the gamma rate for the example dataset is `10110` or `22` in decimal.
- **Epsilon rate**: exactly like the gamma rate, but instead of finding the most common bit, we are asked to find the **least** common bit. In the example, the epsilon rate is `01001` or `9` in decimal.

The result is thus `22 * 9 = 198`.

### Design phase

There are several small challenges to face

1. Convert a binary number to decimal. There's no ready-to-use function in the TensorFlow library, so we have to write it by ourselves.
1. Finding the most common element in a set. A set because the order doesn't matter.

There are also some observations to do:

1. The gamma rate and the epsilon rate are [complementary](https://en.wikipedia.org/wiki/Bitwise_operation#NOT). Hence we can switch from one to the other by applying the bitwise not. e.g `~01001 = 10110`. This means we can focus only on finding the most common element and the least common element can be easily found with a single bitwise operation.
1. When searching for the most frequent elements, we can have undefined situations where the set is perfectly balanced (e.g. 50% of `1` and 50% of `0`). We should handle this undefined situation.

In addition, differently from [day 1](/tensorflow/2021/12/11/advent-of-code-tensorflow/) and [day 2](/tensorflow/2021/12/12/advent-of-code-tensorflow-day-2/), there's no advantage in using a `tf.data.Dataset` object and loop throw it. In fact, the `tf.data.Dataset` object is very convenient when we have to loop over the data, group them, filter them, and apply transformations over it. In this case, however, once we have the input converted it's more convenient to consider the whole dataset as a **single** `tf.Tensor`.

In fact, knowing the cardinality (number of elements) of a TensorFlow dataset is not always possible. For easily addressing consideration 2 expressed above, knowing the cardinality can simplify the problem (we can effectively know if we reached 50%).

### Input pipeline

We create a `tf.data.Dataset` object [as usual](/tensorflow/2021/12/11/advent-of-code-tensorflow/#input-pipeline) but this time, we convert it to a `tf.Tensor` object.

```python
dataset = (
    tf.data.TextLineDataset("input")  # "0101"
    .map(tf.strings.bytes_split)  # '0', '1', '0', '1'
    .map(lambda digit: tf.strings.to_number(digit, out_type=tf.int64))  # 0 1 0 1
)
# We can do this in a raw way, treating the whole dataset as a tensor
# so we can know its shape and extract the most frequent elements easily
tensor_dataset = tf.convert_to_tensor(list(dataset))
```

Interesting is the usage of the `tf.strings.bytes_split` function to convert a string into a tensor of chars and then convert every char into a number.

`tensor_dataset` is a `tf.Tensor` with a shape of `(cardinality, number of bits)`. This representation is very friendly when searching for the most frequent bit.

### Most frequent bits

Using TensorFlow we have the huge advantages of parallel calculation. In fact, instead of singularly looking for the first position, search for the most frequent bit, then move to the second position, search for the most frequent bit, and so on... We can do it all at once.

The most frequent bit, for every position, can be computed as the comparison between the sum of all the elements across the 0-axis and half the dataset cardinality.
We can take into account the undefined scenario (number of ones equal to the number of zero, the equivalent of the number of ones equal to half dataset cardinality), by returning a bitmask-like `tf.Tensor` containing `True` where this condition holds.

```python
@tf.function
def most_frequent_bits(tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Counts the most frequent bits in the input tensor.
    Args:
        tensor: a tensor with shape (cardinality, number of bits)
    Returns:
        (most_frequent, undefined). Both tensor with shape (n_bits).
        - most_frequent: every position contains the most frequent bit
        - undefined: every position containing a True marks that position like undefined.
          There's perfect balanced beweeen 1s and 0s.
    """
    count = tf.reduce_sum(tensor, axis=0)
    tot = tf.cast(tf.shape(tensor)[0], tf.int64)
    half = tot // 2
    ret = tf.cast(tf.greater(count, half), tf.int64)
    return tf.squeeze(ret), tf.squeeze(
        tf.logical_and(tf.equal(count, half), tf.equal(tf.math.mod(tot, 2), 0))
    )  # True where #1 == #0
```

### Binary to decimal conversion

The binary number computed by `most_frequent_bits`, that's a `tf.Tensor` with shape `(n_bits)`, should be converted from binary to decimal to submit (and visualize) the result.

The `bin2dec` function can be easily defined. It's just the implementation, as usual in pure TensorFlow, of the [binary to decimal base conversion](https://en.wikipedia.org/wiki/Binary_number#Decimal).


```python
@tf.function
def bin2dec(bin_tensor: tf.Tensor):
    two = tf.cast(2, tf.int64)
    return tf.reduce_sum(
        tf.reverse(bin_tensor, axis=[0])
        * two ** tf.range(tf.size(bin_tensor), dtype=tf.int64)
    )
```

I defined the `two` as a constant by using the `tf.cast` operation because I wanted to accept as input a `tf.Tensor` with `tf.int64` dtype (I know, a complete waste of storage for only 0s and 1s), and since TensorFlow requires all the types to be the same, I had to explicitly define `two` in this way. The alternative was to define it as a 'tf.constant(2, dtype=tf.int64)`.

### Execution

Since we know, from the initial observations, that `gamma_rate` is the complement of `epsilon_rate` (and vice versa), we already have all we need to solve the problem!

```python
gamma_rate, _ = most_frequent_bits(tensor_dataset)
tf.print("gamma rate (bin): ", gamma_rate)
gamma_rate_dec = bin2dec(gamma_rate)
tf.print("gamma rate (dec): ", gamma_rate_dec)

# epsilon rate is the complement
epsilon_rate = tf.cast(tf.logical_not(tf.cast(gamma_rate, tf.bool)), tf.int64)
tf.print("epsilon rate (bin): ", epsilon_rate)
epsilon_rate_dec = bin2dec(epsilon_rate)
tf.print("epislon rate (dec): ", epsilon_rate_dec)

power_consuption = gamma_rate_dec * epsilon_rate_dec
tf.print("power consumption: ", power_consuption)
```

Here we go, part 1 solved!

As I mentioned in the introduction, part 3 requires the usage of `tf.TensorArray` - let's see where precisely in part 2.

## [Day 3: Binary Diagnostic](https://adventofcode.com/2021/day/3): part two

The puzzle becomes way more complicated. The text asks to determine the so-called **life support rating**, that's the product of the **oxygen generator rating** and **CO2 scrubber rating**.

These 2 rating values are hidden in the input dataset, and there are several conditions for extracting them. I hereby quote the text from the [official Day 3 - Advent of Code 2021] page, since there's no way to summarize the process more than this.

> Both values are located using a similar process that involves filtering out values until only one remains. Before searching for either rating value, start with the full list of binary numbers from your diagnostic report and consider just the first bit of those numbers. Then:
>
> - Keep only numbers selected by the **bit criteria** for the type of rating value for which you are searching. Discard numbers which do not match the bit criteria.
> - If you only have one number left, stop; this is the rating value for which you are searching.
> - Otherwise, repeat the process, considering the next bit to the right.
>
> The bit criteria depends on which type of rating value you want to find:
>
> - To find **oxygen generator rating**, determine the **most common** value (0 or 1) in the current bit position, and keep only numbers with that bit in that position. If 0 and 1 are equally common, keep values with a 1 in the position being considered.
> - To find **CO2 scrubber rating**, determine the **least common** value (0 or 1) in the current bit position, and keep only numbers with that bit in that position. If 0 and 1 are equally common, keep values with a 0 in the position being considered.

The highlighted parts are great hints since they pinpoint some design direction.


### Design phase - part two

The task requires to start considering the first bit. Filter the dataset based on the first bit by satisfying the **bit criteria**, and with this new dataset repeat the process considering the next bit. Repeat until a single value remains.

The bit criteria depend on the rating we are interested in, if oxygen or CO2, and this conditions the usage of our previously defined `most_common_bits`. Moreover, the "undefined" condition we considered earlier will be very handy in this phase given that we need to decide to keep values with a 1 (or 0) in a certain position if the number of 1s and 0s in the examined position matches.

Let's see how to implement the filter by bit criteria.

### Filter by bit criteria

Depending on the criteria, we should produce a new dataset ready for the next iteration. The parameters that change the behavior of the filter are:

- The `current_bit_position` indicates which bit to consider while filtering.
- A boolean flag (called `oxygen`) that changes the behavior from the search from the most common to the least common bit.


```python
class RateFinder(tf.Module):
    def __init__(self, bits):
        super().__init__()
        # Constants
        self._zero = tf.constant(0, tf.int64)
        self._one = tf.constant(1, tf.int64)
        self._two = tf.constant(2, tf.int64)
        # ... we'll add more fields later in the tutorial

    @tf.function(experimental_relax_shapes=True)
    def filter_by_bit_criteria(
        self,
        dataset_tensor: tf.Tensor,
        current_bit_position: tf.Tensor,
        oxygen: tf.Tensor,
    ):
        if oxygen:
            flag = self._one
            frequencies, mask = most_frequent_bits(dataset_tensor)
        else:
            flag = self._zero
            frequencies, mask = most_frequent_bits(dataset_tensor)
            frequencies = tf.cast(
                tf.logical_not(tf.cast(frequencies, tf.bool)),
                tf.int64,
            )
        # #0 == #1 pick the elements with the correct bitflag
        if mask[current_bit_position]:
            indices = tf.where(
                tf.equal(
                    dataset_tensor[:, current_bit_position],
                    flag,
                )
            )
        else:
            indices = tf.where(
                tf.equal(
                    dataset_tensor[:, current_bit_position],
                    frequencies[current_bit_position],
                )
            )

        # All elements with the bit "position" equal to frequencies[position]
        gathered = tf.gather_nd(dataset_tensor, indices)
        return gathered
```

That's the precise implementation of the requirements for the bit criteria. There are 2 details worth mentioning of this implementation

- The constants usage. Instead of using Python numbers, I defined `tf.constant`s in the `init`. This is the recommended approach for having control over the data types, and for avoiding useless conversions. Our graph is really static, and these are constants. Autograph cannot change this.
- `experimental_relax_shapes=True`: the input dataset will change every time we iterate over a new result. `tf.function` by default **creates a new graph** every time the input changes. If you're using Python values, it creates it every time the value changes (hence, never use Python scalars as input!). If you're using (as we are) `tf.Tensor` as input, if the `shape` of the `tf.Tensor` changes, a new graph is created. This behavior is not ideal for this scenario, but luckily we can set the `experimental_relax_shapes` parameter to `True` to change this behavior and re-use the same graph even when the shape changed

Having a tensor that changes its shape is something that may sound strange to many TensorFlow practitioner.

In fact, almost all the objects in TensorFlow are immutable. For example, a `tf.Variable` once defined, can never change its shape. If you try to assign something with a different shape to a `tf.Variable` you'll get an error (and graph-definition time if the shape of the right-hand element is known at that time, at runtime otherwise).

### TensorArray as mutable-shape variables

The only data structure that can change its shape dynamically and be treated (more or less) like `tf.Variable` is [`tf.TensorArray`](https://www.tensorflow.org/api_docs/python/tf/TensorArray?hl=en).

```python
tf.TensorArray(
    dtype, size=None, dynamic_size=None, clear_after_read=None,
    tensor_array_name=None, handle=None, flow=None, infer_shape=True,
    element_shape=None, colocate_with_first_write_call=True, name=None
)
```
The signature is pretty clear. The only required argument is the `dtype`. What's really interesting are the `size` and `dynamic_size` parameters.
The former allows to define the initial size of the TensorArray, the latter enables the TensorArray to grow past its initial size.

This feature seems to perfectly match our requirement of a shape-changing `tf.Tensor` dataset (and not a `tf.data.Dataset` since we do need to know, without looping over the dataset uselessly, the cardinality).

It's possible to read and write singularly the elements of a TensorArray, but for our case, we are interested in the complete read and complete write. For completely overwriting the TensorArray content we need to call the `unstack` method, while for converting the content to a `tf.Tensor` the method to call is `stack`.
This is all we need to know for solving our problem. However, [the documentation](https://www.tensorflow.org/api_docs/python/tf/TensorArray?hl=en) contains lots of examples and info.

#### Finding the ratings

So we have two ratings to find, `oxygen` and `CO2`. We just need to loop over the bits, apply `filter_by_bit_criteria` to obtain a new dataset, and continue until we don't have a single `tf.Tensor` that's our searched rating.

We need some states for our TensorFlow program, one is the `TensorArray` (`_ta`) the other is the rating `_rating`. In particular, we need this `_rating` variable since it's not possible to invoke the `return`  statement while we are inside a loop when working in graph mode.

```python
    def __init__(self, bits):
        # ... previous fields omitted
        self._bits = tf.constant(tf.cast(bits, tf.int64))
        # Variables
        self._rating = tf.Variable(tf.zeros([bits], dtype=tf.int64), trainable=False)
        self._frequencies = tf.Variable(
            tf.zeros([bits], dtype=tf.int64), trainable=False
        )
        self._ta = tf.TensorArray(
            size=1, dtype=tf.int64, dynamic_size=True, clear_after_read=True
        )

    # @tf.function
    def find(self, dataset_tensor: tf.Tensor, oxygen: tf.Tensor):
        num_bits = tf.shape(dataset_tensor)[-1]
        self._ta.unstack(dataset_tensor)
        for current_bit_position in tf.range(num_bits):
            ta = self._ta.stack()
            gathered = tf.squeeze(
                self.filter_by_bit_criteria(ta, current_bit_position, oxygen)
            )
            if tf.equal(tf.size(gathered), num_bits):
                self._rating.assign(gathered)
                break
            self._ta.unstack(gathered)

        return self._rating
```

The `find` method iterates over the `dataset_tensor`, invokes the filter method passing the `oxygen` boolean (tensor) flag and it returns the found rating when found.

### Execution - part two

Very similar to the previous execution, this time we create our `finder` that's an instance of `RateFinder` and use it to find the required rates.

```python
    # gamma_rate contains the most frequent bit in each position 0 1 0 1 0 ...
    # starting from that, we can gather all the numbers that have the more common bit
    # in the "position".
    finder = RateFinder(bits=tf.size(epsilon_rate))

    oxygen_generator_rating = finder.find(tensor_dataset, True)
    tf.print("Oxygen generator rating (bin): ", oxygen_generator_rating)
    oxygen_generator_rating_dec = bin2dec(oxygen_generator_rating)
    tf.print("Oxygen generator rating (dec): ", oxygen_generator_rating_dec)

    co2_generator_rating = finder.find(tensor_dataset, False)
    tf.print("C02 scrubber rating (bin): ", co2_generator_rating)
    co2_generator_rating_dec = bin2dec(co2_generator_rating)
    tf.print("C02 scrubber rating (dec): ", co2_generator_rating_dec)

    tf.print(
        "life support rating = ", oxygen_generator_rating_dec * co2_generator_rating_dec
    )
```

It works! Problem 3 is solved!

But maybe you noticed something strange in the find method...

### TensorArray limitation in graph-mode

Unfortunately, the `find` method can't be converted to its graph counterpart. That's why I commented out the `@tf.function` decoration. There's a known bug/limitation/it-works-in-this-way-by-design in TensorArray that's not clear whenever (if ever) it will be solved.

If I re-enable the decoration, this is what happens

```error
    File "/home/pgaleone/tf/aoc/3/main.py", line 99, in find  *
        self._ta.unstack(gathered)

    ValueError: Cannot infer argument `num` from shape <unknown>
```

In fact, TensorFlow creates static graphs and all this dynamism that TensorArrays give seems to not integrate correctly with the rest of the static-graph ecosystem. The problem is during the `unstack` call, but since `gathered` Tensor shape will change on every iteration this can't be converted to a static graph. :(

## Conclusion

You can see the complete solution in folder `3` on the dedicated Github repository: [https://github.com/galeone/tf-aoc](https://github.com/galeone/tf-aoc).

The challenge has been way more interesting with respect to the day previous two of [day 1](/tensorflow/2021/12/11/advent-of-code-tensorflow/) and [day 2](/tensorflow/2021/12/12/advent-of-code-tensorflow-day-2/). Solving this puzzle allowed me to show how to exploit the native parallelism of TensorFlow for computing reduction operation in parallel over a Tensor.

The second part showed that `TensorArray` is perhaps the more dynamic data structure offered by TensorFlow but it has some problems while working in graph-mode - and that's a pity.

I solved puzzles 4 and 5 and both have been fun. The next articles about my pure TensorFlow solution for day 4 will arrive soon!

For any feedback or comment, please use the Disqus form below - thanks!
