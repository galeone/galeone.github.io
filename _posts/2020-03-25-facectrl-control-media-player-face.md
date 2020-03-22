---
layout: post
title: "FaceCTRL: control your media player with your face"
date: 2020-03-25 12:00:00
categories: tensorflow opencv playerctl
summary: "After being interrupted dozens of times a day while coding with my headphones on, I decided to find a solution that eliminates the stress of pausing and re-playing the song I was listening to. The solution is machine learning application developed with TensorFlow 2, OpenCV, and Playerctl. This article will guide you trough the step required to develop such an application."
authors:
    - pgaleone
---

After being interrupted dozens of times a day while coding with my headphones on, I decided to find a solution that eliminates the stress of pausing and re-playing the song I was listening to.

The idea is trivial:

- When you're in front of your PC with your headphones on: the music plays.
- Someone interrupts you, and you have to remove your headphones: the music pause.
- You walk away from your PC: the music pause.
- You come back to your PC, and you put the headphones on: the music plays again.
- If you want to manually control the player, the manual control has the precedence; e.g. if you have your headphones off and you press play, the music starts as you expect.

The idea is trivial, yes, but implementing it has been fun and challenging at the same time.

In this article I'm going to describe the architecture and the implementation details of this application named [FaceCTRL](https://github.com/galeone/facectrl).

**Note**: if you're not interested in all the implementation details, you can simply go on Github ([galeone/facectrl](https://github.com/galeone/facectrl)) or even just `pip install facectrl` and use it.

## Overview

Below you can see how FaceCTRL works when launched in debug mode:

{:.center}
<iframe width="720" height="480" src="https://www.youtube.com/embed/48N4IU5XB6c" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

In the following we will analyze the steps required to build this application. In particular we will focus on:

- The problem definition
- The technical requirements
- The software architecture
- The machine learning solutions
- Conclusions

Let's start!

## Problem definition

The goal of this application is pretty clear: we have to detect if the person in front of the cameras wears headphones and control the chosen media player depending on the presence/absence of the person and of his/her headphones.

The goal, thus, can be divided in two different parts. The computer vision part that's the part of the software dedicated to the image processing, and the media player part that's dedicated to the control of the media-player. Additionally, The computer vision goal can be divided in 2 separated sub-goals:

1. Detect a face in front of the camera
2. Understand if the person is warning headphones

The problem of detecting a face in an image is a traditional computer vision problem and as such, there exist several face detector algorithm ready to use that works pretty well.

The problem of detecting when a person wears headphones, instead, can be modeled in two different ways:

1. **One class classification / Anomaly detection**. If we follow this path, we must train a model on positive samples only (headphones on), and then use the trained model to detect anomalies.

    - **Pros**: we can detect when the person is wearing headphones.
    - **Cons**: we only know the result is *anomalous* when we detect an anomaly; we don't know if this is because the person has no headphones, or because there is no one in front of the camera.

2. **Binary classification**. We must assume that the only possible outcomes of this problem are two: headphones on and headphones off.

   - **Pros**: we can detect when the person is wearing and not warning headphones.
   - **Cons**: we can't detect anomalous situation: the outcome will always be headphones on/off even if there's no one in front of the camera.

## Requirements

Since the goal is to play and pause the music when someone is in front of the camera we want this application to be *real-time*.

This is a strict requirements and forces us to design our model in order to:

1. Contain few operations
2. Be lightweight (have few parameters)
3. Be in an optimized format designed to speed up the inference

Other than the constraints on the model, we also want to have a lightweight image processing pipeline.

1. Low memory footprint
2. 

## Inference structure




 
## Dataset

The most important part of every machine learning project is the data.

To develop this application the only data we need are pictures of 
