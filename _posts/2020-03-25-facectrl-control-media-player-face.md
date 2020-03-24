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

Let's start by looking at the final result: below you can see how FaceCTRL works when launched in debug mode.

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

From the problem definition, is pretty clear that we want to develop a real time application. This requirement imposes constraints on both the computer vision and media player control parts.

## Software architecture

The software architecture naturally follows from the analysis of the requirements presented above. The diagram below shows the high level architecture, without any implementation detail.

{:.center}
![software high level architecture](/images/facectrl/flowchart.png)

The red blocks are the inputs. In practice, these are implemented as a separate thread that **always** grab frames freely, without waiting for the image processing pipeline.

The green blocks represent the computer vision and machine learning **inference** pipeline.

The yellow blocks represent the media player control pipeline.

To conclude, the blue blocks are the actions: in practice FaceCTRL only have two output actions, play and pause.

Without digging too much into the implementation details (you can have a look at the complete source code [on Github](https://github.com/galeone/facecetrl)), we just want to emphasize that the whole green part is the machine learning **inference** pipeline, and below present some of the building blocks of the architecture.

## Computer Vision pipeline

We want to have a lightweight image processing pipeline, thus we need the face detection and tracking part to have:

1. A fast execution
2. A low memory footprint

At this purpose, OpenCV offers us a whole set of ready to use [Haar Feature-based cascade classifiers](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html) trained to detect faces. These classifiers work on a pyramid of images and thus, the classifiers themselves are capable of working efficiently on images with different resolutions.
OpenCV (**note** not the python module, you need it installed system-wise) comes with this pre-trained models ready to use, we only need to load the parameters from XML files.

We can thus define a `FaceDetector` class that abstract the features we need. The most important part of the code below (file `detector.py`), is the usage of `cv2.CascadeClassifier.detectMultiScale` method, that executes the cascade classification on the image pyramid.

The `crop` method, instead, has the `expansion` parameter that's useful since we're interested in detecting not only the face, but also the headphones. For this reason we want to expand the detected bounding box by a certain amount.

```python
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


class FaceDetector:
    """Initialize a classifier and uses it to detect the bigger
    face present into an image.
    """

    def __init__(self, params: Path) -> None:
        """Initializes the face detector using the specified parameters.

        Args:
            params: the path of the haar cascade classifier XML file.
        """
        self.classifier = cv2.CascadeClassifier(str(params))

    def detect(self, frame: np.array) -> Tuple:
        """Search for faces into the input frame.
        Returns the bounding box containing the bigger (close to the camera)
        detected face, if any.
        When no face is detected, the tuple returned has width and height void.

        Args:
            frame: the BGR input image.
        Returns:
            (x,y,w,h): the bounding box.
        """

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        proposals = np.array(
            self.classifier.detectMultiScale(
                frame,
                scaleFactor=1.5,  # 50%
                # that's big, but we're interested
                # in detecting faces closes to the camera, so
                # this is OK.
                minNeighbors=4,
                # We want at least "minNeighbors" detections
                # around the same face,
                minSize=(frame.shape[0] // 4, frame.shape[1] // 4),
                # Only bigger faces -> we suppose the face to be at least
                # 25% of the content of the input image
                maxSize=(frame.shape[0], frame.shape[1]),
            )
        )

        # If faces have been detected, find the bigger one
        if proposals.size:
            bigger_id = 0
            bigger_area = 0
            for idx, (_, _, width, height) in enumerate(proposals):
                area = width * height
                if area > bigger_area:
                    bigger_id = idx
                    bigger_area = area
            return tuple(proposals[bigger_id])  # (x,y,w,h)
        return (0, 0, 0, 0)

    @staticmethod
    def crop(frame, bounding_box, expansion=(0, 0)) -> np.array:
        """
        Extract from the input frame the content of the bounding_box.
        Applies the required expension to the bounding box.

        Args:
            frame: BGR image
            bounding_box: tuple with format (x,y,w,h)
            expansion: the amount of pixesl the add to increase the
                       bouding box size, from the center.
        Returns:
            cropped: BGR image with size, at least (bounding_box[2], bounding_box[3]).
        """

        x, y, width, height = [
            int(element) for element in bounding_box
        ]  # pylint: disable=invalid-name

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        halfs = (expansion[0] // 2, expansion[1] // 2)
        if width + halfs[0] <= frame.shape[1]:
            width += halfs[0]
        if x - halfs[0] >= 0:
            x -= halfs[0]
        if height + halfs[1] <= frame.shape[0]:
            height += halfs[1]
        if y - halfs[1] >= 0:
            y -= halfs[1]

        image_crop = frame[y : y + height, x : x + width]
        return image_crop
```

Also for the face tracking, OpenCV (**note** in its contrib module) offers a long list of object tracking algorithms already implemented and ready to use. A nice analysis of the trackers has been made by Adrian Rosebrock [here](https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/). We decided to use the CSRT tracking algorithm because it's fast enough and it tracks successfully the face even when rotated.

From the flowchart, it's also clear that the tracker must use a trained classifier and use it to classify the tracked object. In the snippet below you can also see how we handle the tracking failures (the person covers the webcam or walks away) using a threshold; the most important method here is `track_and_classify`.
It's also pretty clear that we abstracted the classifier results using the Enum class `ClassificationResult` (not reported in this article for brevity).

```python
from typing import Tuple

import cv2
import numpy as np

from facectrl.ml import ClassificationResult, Classifier, FaceDetector


class Tracker:
    """Tracks one object. It uses the CSRT tracker."""

    def __init__(
        self, frame, bounding_box, max_failures=10, debug: bool = False,
    ) -> None:
        """Initialize the frame tracker: start tracking the object
        localized into the bounding box in the current frame.
        Args:
            frame: BGR input image
            bounding_box: the bounding box containing the object to track
            max_failures: the number of frame to skip, before raising an
                          exception during the "track" call.
            debug: set to true to enable visual debugging (opencv window)
        Returns:
            None
        """
        self._tracker = cv2.TrackerCSRT_create()
        self._golden_crop = FaceDetector.crop(frame, tuple(bounding_box))
        self._tracker.init(frame, bounding_box)
        self._max_failures = max_failures
        self._failures = 0
        self._debug = debug
        self._classifier = None

    def track(self, frame) -> Tuple[bool, Tuple]:
        """Track the object (selected during the init), in the current frame.
        If the number of attempts of tracking exceed the value of max_failures
        (selected during the init), this function throws a ValueError exception.
        Args:
            frame: BGR input image
        Returns:
            success, bounding_box: a boolean that indicates if the tracking succded
            and a bounding_box containing the tracked objecrt positon.
        """
        return self._tracker.update(frame)

    @property
    def classifier(self) -> Classifier:
        """Get the classifier previousluy set. None otherwise."""
        return self._classifier

    @classifier.setter
    def classifier(self, classifier: Classifier) -> None:
        """
        Args:
            classifier: the Classifier to use
        """
        self._classifier = classifier

    @property
    def max_failures(self) -> int:
        """Get the max_failures value: the number of frame to skip
        before raising an exception during the "track" call."""
        return self._max_failures

    @max_failures.setter
    def max_failures(self, value):
        """Update the max_failures value."""
        self._max_failures = value

    def track_and_classify(
        self, frame: np.array, expansion=(100, 100)
    ) -> ClassificationResult:
        """Track the object (selected during the init), in the current frame.
        If the number of attempts of tracking exceed the value of max_failures
        (selected during the init), this function throws a ValueError exception.
        Args:
            frame: BGR input image
            expansion: expand the ROI around the detected object by this amount
        Return:
            classification_result (ClassificationResult)
        """
        if not self._classifier:
            raise ValueError("You need to set a classifier first.")
        success, bounding_box = self.track(frame)
        classification_result = ClassificationResult.UNKNOWN
        if success:
            self._failures = 0

            crop = FaceDetector.crop(frame, bounding_box, expansion=expansion)
            classification_result = self._classifier(self._classifier.preprocess(crop))[
                0
            ]
        else:
            self._failures += 1
            if self._failures >= self._max_failures:
                raise ValueError(f"Can't find object for {self._max_failures} times")

        return classification_result
```

The third and last part of the image processing pipeline is the classification of the tracked subject. Since we want our model for the classification to be *fast* (since we want to use it on every frame, while tracking) we have to design a model that:

1. Contains few operations.
2. Is lightweight (have few parameters).
3. Can be exported in an optimized format designed to speed up the inference.

Thus, we can try to reduce the number of parameters of the model and reduce the memory footprint assuming that every image is in **gray scale**.

Moreover, since we want to work only with the information contained into the bounding box, we can set a small and fixed input shape for our model. Thus we decide for an input resolution of $$ 64 \times 64 \times 1 $$ for the classification model.

### Media player control

TODO

## The machine learning workflow

The most important part of every machine learning project is the data. Thus, since from the software architecture we know that our classifier should be able to classify the face while being tracked, I good idea is to create a dataset using the tracker itself.

```python
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from facectrl.ml import FaceDetector
from facectrl.video import Tracker, VideoStream


class Builder:
    """Builds the dataset interactively."""

    def __init__(self, dest: Path, params: Path, src: int = 0) -> None:
        """Initializes the dataset builder.

        Args:
            dest: the destination folder for the dataset.
            params: Path of the haar cascade classifier parameters.
            src: the ID of the video stream to use (input of VideoStream).
        Returns:
            None
        """
        self._on_dir = dest / "on"
        self._off_dir = dest / "off"
        if not self._on_dir.exists():
            os.makedirs(self._on_dir)
        if not self._off_dir.exists():
            os.makedirs(self._off_dir)
        self._stream = VideoStream(src)
        self._detector = FaceDetector(params)

    def _acquire(self, path, expansion, prefix) -> None:
        """Acquire and store into path the samples.
        Args:
            path: the path where to store the cropped images.
            expansion: the expansion to apply to the bounding box detected.
            prefix: prefix added to the opencv window
        Returns:
            None
        """
        i = 0
        quit_key = ord("q")
        start = len(list(path.glob("*.png")))
        with self._stream:
            detected = False
            while not detected:
                frame = self._stream.read()
                bounding_box = self._detector.detect(frame)
                detected = bounding_box[-1] != 0

            tracker = Tracker(frame, bounding_box)
            success = True
            while success:
                success, bounding_box = tracker.track(frame)
                if success:
                    bounding_box = np.int32(bounding_box)
                    crop = FaceDetector.crop(frame, bounding_box, expansion)
                    crop_copy = crop.copy()
                    cv2.putText(
                        crop_copy,
                        f"{prefix} {i + 1}",
                        (30, crop.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=2,
                    )
                    cv2.imshow("grab", crop_copy)
                    key = cv2.waitKey(1) & 0xFF
                    if key == quit_key:
                        success = False
                    cv2.imwrite(str(path / Path(str(start + i) + ".png")), crop)
                    i += 1
                frame = self._stream.read()
        cv2.destroyAllWindows()

    def headphones_on(self, expansion=(70, 70)) -> None:
        """Acquire and store the images with the headphones on.
        Args:
            expansion: the expansion to apply to the bounding box detected.
        Returns:
            None
        """
        return self._acquire(self._on_dir, expansion, "ON")

    def headphones_off(self, expansion=(70, 70)) -> None:
        """Acquire and store the images with the headphones off.
        Args:
            expansion: the expansion to apply to the bounding box detected.
        Returns:
            None
        """
        return self._acquire(self._off_dir, expansion, "OFF")
```

Using this class is it possible to create a dataset that contains images like these:

{:.center}
![headphones on](/images/facectrl/on.png)

{:.center}
![headphones on](/images/facectrl/off.png)

