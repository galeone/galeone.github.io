---
layout: post
title: "FaceCTRL: control your media player with your face"
date: 2020-03-25 12:00:00
categories: tensorflow opencv playerctl
summary: "After being interrupted dozens of times a day while coding with my headphones on, I decided to find a solution that eliminates the stress of pausing and re-playing the song I was listening to. The solution is machine learning / computer vision application developed with TensorFlow 2, OpenCV, and Playerctl. This article will guide you trough the step required to develop such an application."
authors:
    - pgaleone
---

After being interrupted dozens of times a day while coding with my headphones on, I decided to find a solution that eliminates the stress of pausing and re-playing the song I was listening to.

The idea is trivial:

- When you're in front of your PC with your headphones on: the music plays.
- Someone interrupts you, and you have to remove your headphones: the music pause.
- You walk away from your PC: the music pause.
- You come back to your PC, and you put the headphones on: the music plays again.
- If you want to manually control the player, the manual control has the precedence; e.g. if you have your headphones off and you press play, the music plays as you expect.

The idea is trivial, yes, but implementing it has been fun and challenging at the same time.

In this article I'm going to describe the software architecture and the implementation details of this application named [FaceCTRL](https://github.com/galeone/facectrl).

**Note**: if you're not interested in all the implementation details, you can simply go on Github ([galeone/facectrl](https://github.com/galeone/facectrl)) or even just `pip install facectrl` and use it.

## Overview

Let's start by looking at the final result: below you can see how FaceCTRL works when launched in debug mode-

{:.center}
<iframe width="720" height="480" src="https://www.youtube.com/embed/48N4IU5XB6c" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

In the following we will analyze the steps required to build this application.

## Problem definition and analysis

The goal of this application is pretty clear: we have to detect if the person in front of the cameras wears headphones and control the chosen media player depending on the presence/absence of the person and of his/her headphones.

The goal, thus, can be divided in two different parts.
The computer vision part dedicated to image processing and understanding, and the media player part dedicated to the control of the media player given the input of the computer vision part.

Focusing on the first part of the problem, we can define a **computer vision pipeline** that works on a video stream. In particular, we expect the pipeline to do:

1. **Face detection**: detect and localize a face in the video stream: we must be sure that someone is in front of the camera.
2. **Face tracking**: the face detection task is usually implemented using neural networks or other traditional methods that, often, are not adequate for a real-time application. Thus, we have to use a tracking algorithm to track the detected face.
3. **Classification**: just after detecting the face, and during the tracking, we have to understand if the person is warning or not headphones.

The problem of detecting a face in an image is a traditional computer vision problem and as such, there exist several face detector algorithm ready to use that works pretty well.
The same reasoning applies to the face tracking, there are several trackers ready to use that are fast, accurate, and robust to subject transformations and occlusions.

The problem of detecting when a person wears headphones, instead, can be modeled in two different ways:

1. **One class classification / Anomaly detection**. If we follow this path, we must train a model on positive samples only (headphones on), and then use the trained model to detect anomalies.

    - **Pros**: we can detect when the person is wearing headphones.
    - **Cons**: we only know the result is *anomalous* when we detect an anomaly; we don't know if this is because the person has no headphones, or because there is no one in front of the camera.

2. **Binary classification**. We must assume that the only possible outcomes of this problem are two: headphones on and headphones off.

   - **Pros**: we can detect when the person is wearing and not warning headphones.
   - **Cons**: we can't detect anomalous situation: the outcome will always be headphones on/off even if there's no one in front of the camera.

In the following we will see how to implement and end-to-end solution to solve this classification problem, from the dataset creation to the model definition, training and inference.

We now have an idea of the problem we want to solve: we know from an high level perspective what's the architecture we expect for the computer vision / machine learning pipeline.

How we should define, instead, the **media player control** part? The second part of the problem, thus, is the development of an application that:

1. Knows how to communicate with a media player: is the media player running?
2. Knows how to play/pause/stop the music.
3. It allows us to receive change of status (e.g. the person while still wearing headphones pressed the pause button - we don't want to automatically restart the music only because he/she still have headphone on)

Moreover, the computer vision pipeline and media player control must work concurrently and communicate any status change almost in real time.

## Technical requirements

From the problem definition, is pretty clear that we want to develop a real time application. This requirement imposes constraints on both the computer vision and media player control parts.

### Computer Vision pipeline

We want to have a lightweight image processing pipeline, thus we need the face detection and tracking part to have:

1. A fast execution
2. A low memory footprint

At this purpose, OpenCV offers us a whole set of ready to use [Haar Feature-based cascade classifiers](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html) trained to detect faces. These classifiers work on a pyramid of images and thus, the classifiers themselves are capable of working efficiently on images with different resolutions.
OpenCV (**note** not the python module, you need it installed system-wise) comes with this pre-trained models ready to use, we only need to load the parameters from XML files.

Also for the face tracking, OpenCV (**note** in its contrib module) offers a long list of object tracking algorithms already implemented and ready to use. A nice analysis of the trackers has been made by Adrian Rosebrock [here](https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/).

The third and last part of the image processing pipeline is the classification of the tracked subject. Since we want our model for the classification to be *fast* (since we want to use it on every frame, while tracking) we have to design a model that:

1. Contains few operations.
2. Is lightweight (have few parameters).
3. Can be exported in an optimized format designed to speed up the inference.

Thus, we can try to reduce the number of parameters of the model and reduce the memory footprint assuming that every image is in **gray scale**.

Moreover, since we want to work only with the information contained into the bounding box, we can set a small and fixed input shape for our model. Thus we decide for an input resolution of $$ 64 \times 64 \times 1 $$ for the classification model.

### Media player control

TODO

## Software architecture

The software architecture naturally follows from the analysis of the requirements presented above. The diagram below shows the high level architecture, without any implementation detail.

{:.center}
![software high level architecture](/images/FaceCTRL.png)

The red blocks are the inputs. In practice, these are implemented as a separate thread that **always** grab frames freely, without waiting for the image processing pipeline.

The green blocks represent the computer vision and machine learning **inference** pipeline.

The yellow blocks represent the media player control pipeline.

To conclude, the blue blocks are the actions: in practice FaceCTRL only have two output actions, play and pause.

Without digging too much into the implementation details (you can have a look at the complete source code [on Github](https://github.com/galeone/facecetrl)), we just want to emphasize that the whole green part is the **inference** pipeline.

Since we decided to use a pre-build face detector and a pre-build tracker, in the next section we'll describe the machine learning workflow followed to develop this solution able to classify the input crop.

## The machine learning workflow

The most important part of every machine learning project is the data.
