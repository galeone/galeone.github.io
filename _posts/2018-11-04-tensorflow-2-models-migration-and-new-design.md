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
- Public 2.0 design process: the community can now work together with the Tensorflow developers and discuss the new features, using the [Tensorflow Discussion Group](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss)
- Compatibility and continuity: a compatibility module with Tensorflow 1.x will be offered, this means that Tensorflow 2.0 will have a module with all the Tensorflow 1.x API inside
- On-disk compatibility: the exported models (checkpoints and frozen models) in Tensorflow 1.x will be compatible with Tensorflow 2.0, only some variable rename could be required
- `tf.contrib`: completely removed. Huge, maintained, modules will be moved to separate repositories; unused and unmaintained modules will be removed.

In practice, if you're new to Tensorflow, you're lucky. If like me, you're using Tensorflow from the 0.x release, you have to rewrite all your codebase (and differently from 0.x to 1.x transition, the changes are massive); however, Tensorflow authors claim that a conversion tool will be released to help the transition. However, conversion tools are not perfect hence manual intervention could be required.

Moreover, you have to change your way of thinking; this can be challenging, but everyone likes challenges, isn't it?

Let's face this challenge and start looking at the changes in detail, starting from the first huge difference: the removal of [`tf.get_variable`](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/get_variable), [`tf.variable_scope`](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/variable_scope), [`tf.layers`](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/layers) and the mandatory transition to a Keras based approach, using [`tf.keras`](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/keras).

Just a note on the release date: it is not defined yet. But from the Tensorflow discussion group, we know that a preview could be released by the end of 2018 and the official release of 2.0 could be in Spring 2019.

Hence is better to update all the existing codebase as soon as the RFCs are accepted in order to have a smooth transition to this new Tensorflow version.

## Keras (OOP) vs Tensorflow 1.x

The [RFC: Variables in TensorFlow 2.0](https://github.com/tensorflow/community/pull/11) has been accepted. This RFC is probably the one with the biggest impact on the existing codebase and requires a new way of thinking for the old Tensorflow users.

As described in the article [Understanding Tensorflow using Go](/tensorflow/go/2017/05/29/understanding-tensorflow-using-go/) every variable has a unique name in the computational graph.

As an early Tensorflow user, I'm used to designing my computational graphs following this pattern:

1. Which operations connect my variable nodes? Define the graph as multiple sub-graphs connected. Define every sub-graph inside a separate [`tf.variable_scope`](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/variable_scope) in order to define the variables of different graphs, inside different scopes and obtain a clear graph representation in [tensorboard](https://twitter.com/paolo_galeone/status/734047400910802944).
2. Do I have to use a sub-graph more than once in the same execution step? Be sure to exploit the [`reuse`](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/variable_scope#__init__) parameter of [`tf.variable_scope`](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/variable_scope) in order to avoid the creation of a new graph, prefixed with `_n`.
3. The graph has been defined? Create the variable initialization op (how many times have you seen the [`tf.global_variables_initializer()`](https://www.tensorflow.org/api_docs/python/tf/initializers/global_variables) call?)
4. Load the graph into a Session and run it.

The example that better shows the reasoning steps, IMHO, is how a simple [GAN](https://en.wikipedia.org/wiki/Generative_adversarial_network) can be implemented in Tensorflow.

### A GAN to understand Tensorflow 1.x

The GAN discriminator $$D$$ must be defined using the [`tf.variable_scope`, `reuse`](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/variable_scope)) parameter, because first we want to feed $$D$$ with real samples, then we want to feed it again with fake samples and only at the end compute the gradient of $$D$$ w.r.t. its parameters.

The generator network $$G$$, instead, is never used twice in the same iteration, hence there's no need to worry about its variables reusing.

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

def discriminator(inputs, reuse=False):
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

This two functions, when called, define inside the default graph 2 different sub-graphs, each one with its own scope ("generator" or "discriminator"). Please note that this function **return the output tensor** of the defined sub-graph, not the graph itself.

In order to share the same $$D$$ graph, we define 2 inputs (real and fake) and define the loss functions required to train $$G$$ and $$D$$.

```python
# Define the real input, a batch of values sampled from the real data
real_input = tf.placeholder(tf.float32, shape=(None,1))
# Define the discriminator network and its parameters
D_real = discriminator(real_input)

# Arbitrary size of the noise prior vector
latent_space_size = 100
# Define the input noise shape and define the generator
input_noise = tf.placeholder(tf.float32, shape=(None,latent_space_size))
G = generator(input_noise)

# now that we have defined the generator output G, we can give it in input to 
# D, this call of `discriminator` will not define a new graph, but it will
# **reuse** the variables previously defined
D_fake = discriminator(G, True)
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

The loss functions are easily defined. The peculiarity of the adversarial training is that first $$D$$ must be trained, using the real samples and the samples generated by $$G$$.
Then, the adversarial, $$G$$, is trained using the result of the $$D$$ evaluation as the input signal.

The adversarial training requires to run **separately** this 2 training steps, but we have defined the models inside the same graph and we don't want to update the $$G$$ variables when we train $$D$$ and vice-versa.

Thus, since we defined **every variable inside the default graph**, hence **every variable is global**, we have to gather the correct variables in 2 different lists and be sure to define the optimizers in order to compute the gradients and apply the updates only to the correct sub-graphs.

```python
# Gather D and G variables
D_vars = tf.trainable_variables(scope="discriminator")
G_vars = tf.trainable_variables(scope="generator")

# Define the optimizers and the train operations
train_D = tf.train.AdamOptimizer(1e-5).minimize(D_loss, var_list=D_vars)
train_G = tf.train.AdamOptimizer(1e-5).minimize(G_loss, var_list=G_vars)
```

Here we go, we're at step 3, graph defined so the last thing to do is to define the variables initialization op:

```python
init_op = tf.global_variables_initializer()
```

#### Pros / Cons

The graph has been correctly defined and, when used inside the training loop and within a session, it works. However, from the software engineering point of view, there are certain peculiarities that are worth noting:

1. The usage of [`tf.variable_scope`](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/variable_scope) context manager to change the (full) name of the variables defined by `tf.layers`: the same call to a `tf.layers.*` method in a different variable scope defines a new set of variables under a new scope.
2. The boolean flag `reuse` can completely change the behavior of any call to a `tf.layers.*` method (define or reuse)
3. Every variable is global: the variables defined by `tf.layers` calling [`tf.get_variable`](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/get_variable) (that's used inside `tf.layers`) are accessible from everywhere: `tf.trainable_variables(prefix)` used above to gather the 2 lists of variables perfectly describes this.
4. Defining sub-graphs is not easy: you just can't call `discriminator` and get a new, completely independent, discriminator. Is a little bit counterintuitive.
5. The return value of a sub-graph definition (call to `generator`/`discriminator`) is only its output tensor and not something with all the graph information inside (although is possible to backtrack to the input, but it's not that easy)
6. Defining the variables initialization op is just boring (but this is just been resolved using [`tf.train.MonitoredSession`](https://www.tensorflow.org/api_docs/python/tf/train/MonitoredSession) and [`tf.train.MonitoredTrainingSession`](https://www.tensorflow.org/api_docs/python/tf/train/MonitoredTrainingSession). *hint: use them*.)

Those 6 points are probably all *cons*.

We defined our GAN in the Tensorflow 1.x way: let's start the migration to Tensorflow 2.0

### A GAN to understand Tensorflow 2.x

As stated in the previous section, in Tensorflow 2.x, the way of thinking changes. The removal of [`tf.get_variable`](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/get_variable), [`tf.variable_scope`](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/variable_scope), [`tf.layers`](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/layers) and the mandatory transition to a Keras based approach, using [`tf.keras`](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/keras) forces the Tensorflow developer to change its mindset.

We have to define the generator $$G$$ and discriminator $$D$$ using `tf.keras`: this will give us for free the variable sharing feature that we used to define $$D$$, but implemented differently under the hood.

Please note: `tf.layers` will be **removed**, hence starting to use `tf.keras` right now to define your models is mandatory in order to be ready to 2.x.

```python
def generator(input_shape):
    """generator network.
    Args:
        input_shape: the desired input shape (e.g.: (latent_space_size))
    Returns:
        G: The generator model
    """
    inputs = tf.keras.layers.Input(input_shape)
    net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name="fc1")(inputs)
    net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name="fc2")(net)
    net = tf.keras.layers.Dense(units=1, name="G")(net)
    G = tf.keras.Model(inputs=inputs, outputs=net)
    return G

def discriminator(input_shape):
    """discriminator network.
    Args:
        input_shape: the desired input shape (e.g.: (latent_space_size))
    Returns:
        D: the discriminator model
    """
    inputs = tf.keras.layers.Input(input_shape)
    net = tf.keras.layers.Dense(units=32, activation=tf.nn.elu, name="fc1")(inputs)
    net = tf.keras.layers.Dense(units=1, name="D")(net)
    D = tf.keras.Model(inputs=inputs, outputs=net)
    return D
```

Look at the different approach: both `generator` and `discriminator` returns a `tf.keras.Model` and not just an output tensor.

This means that using Keras we can instantiate our model and use **the same model** in different parts of the source code and we effectively use the variables of that model, without the problem of defining a new sub-graph prefixed with `_n`. In fact, differently from the 1.x version, we're going to define just one $$D$$ model and use it twice.

```python
# Define the real input, a batch of values sampled from the real data 
real_input = tf.placeholder(tf.float32, shape=(None,1))

# Define the discriminator model
D = discriminator(real_input.shape[1:])

# Arbitrary set the shape of the noise prior vector
latent_space_size = 100
# Define the input noise shape and define the generator
input_noise = tf.placeholder(tf.float32, shape=(None,latent_space_size))
G = generator(input_noise.shape[1:])
```

Again: there's no need to define `D_fake` as we did above, and there's no need to think ahead when defining the graphs and worry about the variable sharing.

Now we can go on and define the $$G$$ and $$D$$ loss functions:

```python
D_real = D(real_input)
D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))
)

G_z = G(input_noise)

D_fake = D(G_z)
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake))
)

D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake))
)
```

So far so good. The last thing to do is defining the 2 optimizers that will optimize $$D$$ and $$G$$ separately. Since we're using `tf.keras` there's no need to manually create the list of the variables to update, since are the `tf.keras.Model`s objects themselves that are carrying this attribute:

```python
# Define the optimizers and the train operations
train_D = tf.train.AdamOptimizer(1e-5).minimize(D_loss, var_list=D.trainable_variables)
train_G = tf.train.AdamOptimizer(1e-5).minimize(G_loss, var_list=G.trainable_variables)
```

We're ready to go: we reached step 3 and since we're still working using the static graph mode, we have to define the variables initialization op:

```python
init_op = tf.global_variables_initializer()
```

### Pros/Cons

- Transitioning from `tf.layers` to `tf.keras` it easy: all `tf.layers` methods have their own `tf.keras.layers` counterpart
- `tf.keras.Model` completely removes to worry about variables reusing, issues on graph redefinition
- `tf.keras.Model` is not an output tensor, but is a complete model that carries its own variables
- We still have to initialize all variables, but as said before `tf.train.MonitoredSession` can do it for us

The GAN example, in both Tensorflow 1.x and 2.x, has been developed using the "old" paradigm of graph definition first, execution in a session next (that is and will be a good and valid paradigm to follow and - personal opinion - is the best one).

However, another big change in Tensorflow 2.x is to make the *eager mode* the default execution mode. In Tensorflow 1.x we have to explicitly enable the eager execution, while in Tensorflow 1.x we'll have to do the opposite.

## Eager mode first



The transition to Tensorflow 2.x carries other changes that I tried to summarize in the next *what if* section.

## What if?

The following is a list of what I think will be the F.A.Q. about the transition to Tensorflow 2.x.

##### What if my project uses `tf.contrib`?

All the information about the fate of the projects inside `tf.contrib` can be found here: [Sunsetting tf.contrib](https://github.com/tensorflow/community/blob/rfc-contrib/rfcs/20180907-contrib-sunset.md).

Probably you just have to `pip install` a new python package or rename your `tf.contrib.something` to `tf.something`.

##### What if a project working in Tensorflow 1.x stops working in 2.x?

This shouldn't happen: please double check that the transition has been correctly implemented and if it is, open a bug report on GitHub.

##### What if a project works in static graph mode but it doesn't in eager mode?

That's a problem I'm currently facing, as I reported here: [Tensorflow eager version fails, while Tensorflow static graph works](https://github.com/tensorflow/tensorflow/issues/23407).

Right now I don't know if this is a bug from my side or there's something wrong in the actual Tensorflow eager version. However, since I'm used to thinking in a static graph oriented way, I'll just avoid using the eager version.

##### What if a method from `tf.` disappeared in 2.x?

There's a high chance the method has only been moved. In Tensorflow 1.x there are a lot of aliases for a lot of methods, in Tensorflow 2.x instead, there's the aim (if the [RFC: TensorFlow Namespaces](https://github.com/tensorflow/community/blob/25ab399ecf66f7cee8e7f8c479aefcb96f8cc96b/rfcs/20180827-api-names.md) will be accepted - as I wish) of removing a lot of these aliases and move methods to a better location, in order to increase the overall coherence.

In the RFC you can find the newly proposed namespaces, the list of the one that will be removed and all the other changes that (probably) will be made to increase the coherence of the framework.

Also, the conversion tool that will be released will be probably able to correctly apply all these updates for you (this is just my speculation on the conversion tool, but since it's an easy task that's probably a feature that will be present).

## Conclusion

This article has been created with the specific aim of shed a light on the changes and the challenges that Tensorflow 2.0 will bring to us, the framework users.

The GAN implementation in Tensorflow 1.x and its conversion in Tensorflow 2.x should be a clear example of the mindset change required to work with the new version.

Overall I think Tensorflow 2.x will improve the quality of the framework and it will standardize and simplifies how to use it.

There are certain parts of the update that I don't like, but are just my personal opinions:

- The focus on the eager execution and make it the default: it looks too much a marketing move to me. It looks like Tensorflow wants to chase PyTorch (eager by default)
- Switching to a Keras based approach is a good move, but it makes the graph visualized in Tensorboard *really* ugly. In fact, the variables and the graphs are defined globally, and the `tf.named_scope` (invoked every time a Keras Model is called, in order to share the variables easily) that creates a new "block" in the tensorflow graph, is separated by the graph it uses internally and it has in the list of the input nodes all the variables of the model - this makes the graph visualization of tensorboard pretty much useless and that's a pity for such a good tool.

If you liked the article feel free to share it using the buttons below and don't hesitate to comment to let me know if there's something wrong/that can be improved in the article.

Thanks for reading!
