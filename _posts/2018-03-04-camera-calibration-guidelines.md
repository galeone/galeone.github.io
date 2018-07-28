---
layout: post
title: "Camera calibration guidelines"
date: 2018-03-04 08:00:00
categories: computer-vision
summary: "The process of geometric camera calibration (camera resectioning) is a fundamental step for machine vision and robotics applications. Unfortunately, the result of the calibration process can vary a lot depending on various factors. There are a lot of empirical guidelines that have to be followed in order to achieve good results: this post will drive you through them."
---

The process of geometric camera calibration (camera resectioning) is a fundamental step for machine vision and robotics applications. Unfortunately, the result of the calibration process can vary a lot depending on various factors. There are a lot of empirical guidelines that have to be followed in order to achieve good results: this post will drive you through them.

If you're already familiar with the camera calibration concept and you don't want to read a brief summary again, you can just jump to the [guideline section](#guidelines).

## Camera Calibration

If you're reading this post, you already know what camera calibration is and why you need it, hence I won't dig into this topic too much. Let's just summarize the whole process in a very coarse way:

Camera calibration is needed when:

1. You're developing a machine vision application (measuring objects) and therefore a good estimation of the camera parameters is required to to correctly measure the planar objects
2. The images grabbed are affected by radial and/or tangential distortion and we want to remove them
3. You just want to measure the amount of distortion introduced by the lenses
4. Any other reason

Whatever the reason is the aim is the same: estimate in the most accurate way the parameters of a pinhole camera model that approximates the camera that produced the images.
In particular, we're interested in the most accurate estimation of the distortion coefficients and the camera matrix:

$$ \begin{align}
\text{distortion coefficients} &= (k_1 \hspace{10pt} k_2 \hspace{10pt} p_1 \hspace{10pt} p_2 \hspace{10pt} k_3) \\
\text{camera matrix} &= \left ( \begin{matrix} f_x & \alpha & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{matrix} \right ) \\
\end{align} $$

The distortion parameters came out from the non-linear equation of the radial (1) and tangential (2) distortion that maps distorted points coordinates to the corresponding undistorted coordinates (the hat indicates distortion):

$$ \begin{align}
1: \begin{cases}
\hat{x} &= x( 1 + k_1 r^2 + k_2 r^4 + k_3 r^6) \\
\hat{y} &= y( 1 + k_1 r^2 + k_2 r^4 + k_3 r^6) \\
    \end{cases} & \;
2: \begin{cases}
\hat{x} &= x +  2p_1xy + p_2(r^2+2x^2) \\
\hat{y} &= y +  p_1(r^2+ 2y^2)+ 2p_2xy 
\end{cases} 
\end{align} $$

while the camera matrix parameters capture the relation between the points in the scene (3D world) and the corresponding points in the image (2D world). In particular, the camera matrix contains the following parameters:

1. $$ f_x, f_y $$ focal lengths in pixel along the $$x$$ and $$y$$ axis
2. $$ \alpha $$ the skew coefficient that defines the angle between the x and y planes (usually, but not always, is 0)
3. $$ c_x, c_y $$ pixel coordinates of the principal point

The calibration process, of course, allows to estimate all this parameters. The most widely used camera calibration algorithm is the Zhang algorithm [^1] that iteratively finds correspondences between the coordinates of some easily identifiable feature points (usually inner corners) of a known object (a planar pattern) in both the image plane and the scene.

The algorithm iteratively builds and refines a system of equations (equations that are the mapping between the points in the image and the scene) with the aim of minimizing the reprojection error.

The reprojection error is the total sum of squared distances between the observed feature points and the projected object points.

Now that we have recalled the basics of the camera calibration we can go through the guidelines required to achieve a good camera calibration.

{% include inarticlead.html %}

## Guidelines

Before going trough the guidelines for the calibration it is worth to spend a few words on the camera physical setup. If you're developing a machine vision application, in which the camera must remain fixed and the objects to measure can span the FOV, it's worth to first do the camera setup (hence do the math for the working distance calculation giving your lenses, the sensor size, the accuracy required, ecc... hint: there's a good tool by Basler that can help you doing the setup [^2]) and once the camera is in its working place start with the calibration.
However, no matter what your application is, the following guidelines holds.

### Camera auto-focus

If your camera has some sort of auto-focus: **disable it**. Auto-focus changes the position of the lenses dynamically making your calibration procedure completely useless.

### Camera manual focus

If you're developing some machine vision application, first complete the camera setup, then use a viewer software viewer to see the captured images from the camera and manually adjust the focus of your camera until the subjects are perfectly in focus.

Use the locating skew to lock in the desired position the focus ring. Never touch the skew or the focus ring again.

### Vibrations and blur

Mechanically lock the camera. Keep the vibrations of the camera the lowest possible. The quality of the captured images depends a lot on the blur introduced by the vibrations, keeping the vibrations low reduces the blur and increase the quality of the image, making the feature points detection easy and accurate.

When taking the picture of the planar pattern, try to keep the motion blur low.

### Pattern quality

The pattern size and quality is of extreme importance. Let's consider the case of a chessboard pattern. The calibration process requires to detect the inner corner of the chessboard and the assumption of the algorithm is that every chessboard square is a perfect square. Another assumption is, hence, that the pattern is perfectly planar. The respect of these assumptions is not trivial.

To respect the assumptions, the pattern must follow this rules:

1. **Pattern size**: the pattern size should be proportional to the FOV size. The higher the number of inner corners the better (an asymmetrical $$20 \times 19$$ chessboard is usually good. But everything depends on the FOV size).
2. **Pattern borders**: finding the inner corners of the chessboard requires to first identify the chessboard. The white margins around the chessboard allows the corner detection algorithm to work well and faster (intuitively: it's easier to find the chessboard and its corners when the chessboard itself is easily identifiable thanks to the white margins). A white margin as big as the chessboard side is usually OK.
3. **Pattern manufacturing quality**: do not print the pattern at home. Seriously. Go to a professional print shop and ask them to create the pattern for you. They have the right software to create the chessboard with squares with the real desired square size with an extremely high accuracy. More than everything, they can print the pattern on a **white opaque, rigid and planar** material. Stress the print shop to print on some extremely rigid opaque white material.

    In short: choose a big pattern with a good number of inner corners and a reasonable thick white border. Do not print the pattern at home. Do not make it print on paper or any other material that's not rigid and opaque.

{% include inarticlead.html %}

### Taking pictures

Once you got an high quality pattern, you have to take the pictures that the calibration algorithm will use. This is one of the most neglected part of the calibration process but its somehow the most important one.

1. Move the pattern **everywhere** in the calibration volume.
2. Take pictures of the pattern in every possible position.
3. **Angle the pattern with respect to the focal axis** and move the inclined pattern all over the calibration volume.
4. Uniformly distribute the measurements in the calibration volume.

The calibration volume is a cone-shaped volume whose base is the FOV and the height is the working distance.

The number of images to use for the calibration process depends on the size of the pattern, the number of inners corners and the amount of volume filled.
Ideally, if you have a big pattern, with "a lot" of inner corners and you took pictures of the pattern with different angle in every part of the calibration volume you'll be fine.

However, a rule of thumb could be: use about 50 **different** images of the calibration pattern.

### Analyzing pictures

The calibration algorithm does some preprocessing of the images before running the corner detection algorithm. Check out of in your specific case some of the preprocessing steps will led to worse results.

For example, in OpenCV there's the `flag` parameter of the `cv::findChessboardCorners` method that has the default value of `CALIB_CB_ADAPTIVE_THRESH+CALIB_CB_NORMALIZE_IMAGE`.

Check out if in your specific case this 2 preprocessing steps will bring to some good images with the chessboard highlighted or there's some lighting issue that the histogram equalization will enhance making the Otsu algorithm binarize the image in the wrong way.

This step will also help you to find issues in the lights setup and to correct them.

### Use the right model for your camera

The camera calibration algorithm is extremely general. It allows to estimate any parameter described in the previous section. However, not always is required to estimate everything because you're trying to model a real system and the system can already have some constraint that make parameters constants.

1. **Aspect ratio**: the camera matrix has two different focal lengths: $$f_x$$ and $$f_y$$. The ratio between them is called aspect ratio and describes the amount of deviation from a perfectly square pixel. Hence, go to the datasheet of your camera and check the pixel width and height. If the pixel is a perfect square, you can remove a degree of freedom from the problem, forcing the constraint $$f_x = f_y = f$$.
2. **Skew**: the $$\alpha$$ parameter measure the skew between the x and y axis of the image plane. It very unlikely in today cameras that this parameter is different from zero. Hence, if you have the camera datasheet verify if there's skew and if present just fix the value to the specified amount. Otherwise set it to zero.
3. **Principal point**: if you have a good reason to think that the lenses are off-center with respect to the image, then try to estimate it. Otherwise, that's the most common case, just set it to the center of the image.


In short: you're modelling a real system, look at the reality and do not add useless constraints or free parameters.

### Results

Once you have correctly created the pattern, set up the camera, taken the pictures and run the calibration algorithm you have to evaluate the results.

The aim of the algorithm is to minimize the reprojection error, hence we do expect to have a RMS value (in px) lower than 1 (I'd say, a value >= 1 px is unacceptable. A value < 1 start being OK. A value < 0.5 is good).

However, a low RMS value is only a necessary but not sufficient condition for a good calibration. In fact, low RMS simply means that the algorithm was able to minimize the objective function on the given input images **on average**. Instead, we're interested on the global results, image per image, corner to corner.

A good approach is to manually measure for every inner corner the reprojection error and plot the results in a histogram. Ideally, we want an histogram completely centered at zero. If the histogram is bimodal or has some peak far from zero, then there's something extremely wrong in the calibration and everything should be checked again and the calibration procedure repeated.

### Conclusion

Camera calibration is a fundamental step in many computer vision applications. Too often the empirical guidelines have to be learned the hard way (via a lot of trial and error) and a complete list do not exists... until now.

The aim of this article is to give a set of guidelines to follow about the camera calibration, the pattern quality manufacturing and how to check if the calibration reached is aim or the results are biased.

If you know other tips for reaching a good camera calibration feel free to contribute to this article leaving a comment in the section below or opening an issue on GitHub to discuss your suggestion and than open a PR for adding your contribution to the article.

---
[^1]: Zhang. A Flexible New Technique for Camera Calibration. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11):1330-1334, 2000.
[^2]: https://www.baslerweb.com/en/products/tools/lens-selector/
