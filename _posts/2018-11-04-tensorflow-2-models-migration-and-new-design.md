---
layout: post
title: "Tensorflow 2.0: models migration and new design"
date: 2018-11-01 08:00:00
categories: tensorflow gan
summary: "Tensorflow 2.0 will be a major milestone for the most popular machine learning framework: lots of changes are coming, and all with the aim of making ML accessible to everyone. These changes, however, requires for the old users to completely re-learn how to use the framework: this article describes all the (known) differences between the 1.x and 2.x version, focusing on the change of mindset required and highlighting the pros and cons of the new implementation."
---

Tensorflow 2.0 will be a major milestone for the most popular machine learning framework: lots of changes are coming, and all with the aim of making ML accessible to everyone. These changes, however, requires for the old users to completely re-learn how to use the framework: this article describes all the (known) differences between the 1.x and 2.x version, focusing on the change of mindset required and highlighting the pros and cons of the new and implementations.

This article can be a good starting point also for the novice: start thinking in the Tensorflow 2.0 way right now, so you don't have to re-learn a new framework (unless until Tensorflow 3.0 will be released).

## Tensorflow 2.0: why and when?

The idea is to make Tensorflow easier to learn and apply.

The first glimpse on what Tensorlow 2.0 will be has been given by Martin Wicke, one of the Google Brain Engineers, in the [Announcements Mailing List](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce), [here](https://groups.google.com/a/tensorflow.org/forum/#!topic/announce/qXfsxr2sF-0). In short:

- Eager execution will be a central feature of 2.0. It aligns users' expectations about the programming model better with TensorFlow practice and should make TensorFlow easier to learn and apply.
- Support for more platforms and languages, and improved compatibility and parity between these components via standardization on exchange formats and alignment of APIs.
- Remove deprecated APIs and reduce the amount of duplication, which has caused confusion for users.
- Public 2.0 design process: the community can now work together with the Tensorflow developers and discuss about the new features, using the [Tensorflow Discussion Group](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss)
- Compatibility and continuity: a compatibility module with Tensorflow 1.x will be offered, this means that Tensorflow 2.0 will have a module with all the Tensorflow 1.x API inside
- On-disk compatibility: the exported models (checkpoints and frozen models) in Tensorflow 1.x will be compatible for the usage in Tensorflow 2.0, only some variable rename could be required
- `tf.contrib`: completely removed. Huge, maintained, modules will be moved to separate repositories; unused and unmaintaned modules will be removed.

In practice, if you're new to Tensorflow, you're lucky. If, like me, you're using Tensorflow from the 0.x release, you have to rewrite all your codebase (and differently from 0.x to 1.x transition, the changes are massive); however, they claim that a conversion tool will be released. However, conversion tools are not perfect, hence manual intervention could be required.

Moreover, you have to change your way of thinking; this can be challenging, but everyone likes challenges, isn't it?

Let's face this challenge and start looking at the changes in detail, starting from the first huge difference: the removal of `tf.get_variable`, `tf.variable_scope`, `tf.layers` and the mandatory transition to a Keras based approach.

Just a note on the release date: it is not defined yet. But from Tensorflow discussion group, we know that a preview could be released by the end of 2018 and the official release of 2.0 could be in Spring 2019.

## Keras (OOP) vs Tensorflow 1.x

The [RFC: Variables in TensorFlow 2.0](https://github.com/tensorflow/community/pull/11) has been accepted. This RFC is probably the one with the biggest impact on the existing codebase and requires a new way of thinking for the old Tensorflow users.

As described in the article [Understanding Tensorflow using Go](/tensorflow/go/2017/05/29/understanding-tensorflow-using-go/) every variable has a unique name in the computational graph.

As a Tensorflow 1.x user, I'm used to think following this pattern

1. Which operations connect my variable nodes? Define the graph as multiple sub-graphs connected. Define every sub-graph inside a separate [`tf.variable_scope`](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/variable_scope) in order to define the variables of different graphs, inside different scopes and obtain a clear graph representation in tesorboard.
2. I have to use a sub-graph more then once in the same execution step? Be sure to exploit the `reuse` parameter of [`tf.variable_scope`](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/variable_scope) in order to avoid the creation of a new graph, prefixed with `_n`.
3. The graph has been defined? Create the variable initialization op (how many times have you seen the `tf.global_variables_initializer()` call?)
4. Load the graph into a Session and run it.

The example that better shows this reasoning, is a simple [GAN](https://en.wikipedia.org/wiki/Generative_adversarial_network).

### A GAN to understand Variables in Tensorflow 1.x

The GAN discriminator $$D$$ must be defined using the `tf.variable_scope` `reuse` parameter, because first we want to feed $$D$$ with real samples, then we want to feed it again but with fake samples and only at the end compute the gradient of $$D$$ w.r.t. it's parameters.

The generator network $$G$$, instead, is never used twice in the same iteration, hence its variables are never reused.

```python
def generator(inputs):
    """generator network.
    Args:
        inputs: a (None, latent_space_size) tf.float32 tensor
    Returns:
        G: the generator output node
    """
    with tf.variable_scope("generator"):
        fc1 = tf.layers.dense(inputs, units=64, activation=tf.nn.elu, name="fc1")
        fc2 = tf.layers.dense(fc1, units=64, activation=tf.nn.elu, name="fc2")
        G = tf.layers.dense(fc1, units=1, name="G")
    return G

def disciminator(inputs, reuse=False):
    """discriminator network
    Args:
        inputs: a (None, 1) tf.float32 tensor
        reuse: python boolean, if we expect to reuse (True) or declare (False) the variables
    Returns:
        D: the discriminator output node
    """
    with tf.variable_scope("discriminator", reuse=reuse):
        fc1 = tf.layers.dense(inputs, units=32, activation=tf.nn.elu, name="fc1")
        D = tf.layers.dense(fc1, units=1, name="D")
    return D
```

This two functions, when called, define inside the global graph 2 different sub-graphs, each one in its own scope ("generator" or "discriminator"). Please not that this function **return the output tensor** of the defined sub-graph, not the graph itself.

In order to share the same $$D$$ graph, we define 2 inputs (real and fake) an define the loss function required to train $$G$$ and $$D$$.

```python
# Define the real input, a batch of values sampled from the real data
real_input = tf.placeholder(tf.float32, shape=(None,1))
# Define the discriminator network and its parameters
D_real = disciminator(real_input)

# Arbitrary size of the noise prior vector
latent_space_size = 100
# Define the input noise shape and define the generator
input_noise = tf.placeholder(tf.float32, shape=(None,latent_space_size))
G = generator(input_noise)

# now that we have defined the generator output G, we can give it in input to 
# D, this call of `discriminator` will not define a new graph, but it will
# **reuse** the variables previously defined
D_fake = disciminator(G, True)
```

The last thing to do is to just define the 2 loss functions and the 2 optimizers required to train $$D$$ and $$G$$ respectively.

```python
D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))
)

D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake))
)

# D_loss, when invoked it first does a forward pass using the D_loss_real
# then another forward pass using D_loss_fake, sharing the same D parameters.
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake))
)
```

The loss functions are easily defined. The peculiarity of the adversarial training is that first $D$ is trained and only after $G$ is trained, using the result of the $$D$$ evaluation. Hence we need to run **separately** this 2 training steps, but we have defined the models inside the same graph and we don't want to update the $$G$$ variables when we train $$D$$ and vice-versa.

Thus, since we defined **every variable inside the global graph**, hence **every variable is global**, we have to gather the variables in 2 lists and be sure to define the optimizers in order to compute the gradients and apply the updates, only of the correct sub-graphs.

```python
# Gather D and G variables
D_vars = tf.trainable_variables(scope="discriminator")
G_vars = tf.trainable_variables(scope="generator")

# Define the optimizers and the train operations
train_D = tf.train.AdamOptimizer(1e-5).minimize(D_loss, var_list=D_vars)
train_G = tf.train.AdamOptimizer(1e-5).minimize(G_loss, var_list=G_vars)
```

Here we go, we're at the step 3, graph defined so the last thing to do is to define the variables initialization op:

```python
init_op = tf.global_variables_initializer()
```

#### Pros / Cons

The graph has been correctly defined and, when used inside the training loop and within a session, it works. However, from the software engineering point of view, there are certain peculiarities that's worth noting:

1. The usage of `tf.variable_scope` to change the (complete) name of the variables defined by `tf.layers`: the same call to a `tf.layer.*` method, in a different variable scope, defines a new set of variables under a new scope.
2. The boolean flag `reuse` of the `tf.variable_scope` context manager can completely change the behavior of the call to `tf.layer` (define or reuse)
3. Every variable is global: the variables defined by `tf.layers` calling `tf.get_variable` (that's used inside `tf.layers`) are accessible from everywhere: `tf.trainable_variables(prexix)` perfectly describes this.
4. Defining sub-graphs is not easy: you just can't call `discriminator` and get a new, completely independent, discriminator.
5. The return value of a sub-graph definition (call to `generator`/`discriminator`) is only its output tensor and not something with all the graph information inside (although is possible to backtrack to the input, but it's not that easy)
