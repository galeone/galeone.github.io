---
layout: post
title: "Tensorflow 2.0: models migration and new design"
date: 2018-11-01 08:00:00
categories: tensorflow
summary: "Tensorflow 2.0 will be a major milestone for the most popular machine learning framework: lots of changes are coming, and all with the aim of making ML accessible to everyone. These changes, however, requires for the old users to completely re-learn how to use the framework: this article describes all the (known) differences between the 1.x and 2.x version, focusing on the change of mindset required and highlighting the pros and cons of the new implementation."
---

Tensorflow 2.0 will be a major milestone for the most popular machine learning framework: lots of changes are coming, and all with the aim of making ML accessible to everyone. These changes, however, requires for the old users to completely re-learn how to use the framework: this article describes all the (known) differences between the 1.x and 2.x version, focusing on the change of mindset required and highlighting the pros and cons of the new implementation.

This article can be a good starting point also for the novice: start thinking in the Tensorflow 2.0 way right now, so you don't have to re-learn a new framework (unless until Tensorflow 3.0 will be released).

## Tensorflow 2.0: why?

The idea is to make Tensorflow easier to learn and apply.

The first glimpse on what Tensorlow 2.0 will be has been given by Martin Wicke, one of the Google Brain Engineers, in the [Announcements Mailing List](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce), [here](https://groups.google.com/a/tensorflow.org/forum/#!topic/announce/qXfsxr2sF-0). In short:

- Eager execution will be a central feature of 2.0. It aligns users' expectations about the programming model better with TensorFlow practice and should make TensorFlow easier to learn and apply.
- Support for more platforms and languages, and improved compatibility and parity between these components via standardization on exchange formats and alignment of APIs.
- Remove deprecated APIs and reduce the amount of duplication, which has caused confusion for users.
- Public 2.0 design process: the community can now work together with the Tensorflow developers and discuss about the new features, using the [Tensorflow Discussion Group](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss)
- Compatibility and continuity: a compatibility module with Tensorflow 1.x will be offered, this means that Tensorflow 2.0 will have a module with all the Tensorflow 1.x API inside
- On-disk compatibility: the exported models (checkpoints and frozen models) in Tensorflow 1.x will be compatible for the usage in Tensorflow 2.0, only some variable rename could be required
- `tf.contrib`: completely removed. Huge, maintained, modules will be moved to separate repositories; unused and unmaintaned modules will be removed.

In practice, if you're new to Tensorflow, you're lucky. If, like me, you're using Tensorflow from the 0.x release, you have to rewrite all your codebase (and differently from 0.x to 1.x transition, the changes are massive) and you have to change your way of thinking; this can be challenging, but everyone likes challenges, isn't it?

Let's face this challenge and start looking at the changes in detail, starting from the first huge difference: the removal of `tf.get_variable`, `tf.variable_scope`, `tf.layers` and the mandatory transition to a Keras based approach.

### Keras (OOP) vs Tensorflow 1.x
