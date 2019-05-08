import tensorflow as tf


def if_elif_eager(a, b):
    if a > b:
        tf.print("a > b", a, b)
    elif a == b:
        tf.print("a == b", a, b)
    else:
        tf.print("a < b", a, b)


x = tf.constant(1)
if_elif_eager(1, x)


@tf.function
def if_elif_fixed(a, b):
    if tf.math.greater(a, b):
        tf.print("a > b", a, b)
    elif tf.math.equal(a, b):
        tf.print("a == b", a, b)
    else:
        tf.print("a < b", a, b)


print("fixed")
print(tf.autograph.to_code(if_elif_fixed.python_function))
if_elif_fixed(1, x)
