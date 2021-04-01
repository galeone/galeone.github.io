---
layout: post
title: "Creating TensorFlow Custom Ops, Bazel, and ABI compatibility"
date: 2021-04-01 08:00:00
categories: tensorflow bazel abi c++
summary: "Custom ops are a way for extending the TensorFlow framework by adding operations that are not natively available in the framework. Adding a new operation is a relatively simple thing especially if you work in the officially supported environment (Ubuntu16, CUDA 10). However, if you built TensorFlow from scratch to support your target environment (e.g. Archlinux, CUDA 11) the official TensorFlow support for creating a custom op - that relies upon a Docker image - becomes useless."
authors:
    - pgaleone
---

Custom ops are a way for extending the TensorFlow framework by adding operations that are not natively available in the framework. Adding a new operation is a relatively simple thing especially if you work in the officially supported environment (Ubuntu16, CUDA 10). However, if you built TensorFlow from scratch to support your target environment (e.g. Archlinux, CUDA 11) the official TensorFlow support for creating a custom op - that relies upon a Docker image - becomes useless.

This article will guide you through the steps required to build TensorFlow from source and create a custom op. Moreover, the conversion process required to adapt a "custom op" (designed to be created with the Docker image), and a "user op" (an operation placed inside the TensorFlow source code and build with Bazel) is presented. So, we'll see in order:

- Building TensorFlow from source.
- Custom Ops overview.
- Custom Ops with GPU support.
- Adapting a custom op to a user op - the TensorFlow 3D use case.

During this process, we'll discover Bazel and slowly dig into what I call the Bazel Hell.

## Building TensorFlow from source

The target platform for this build is my current setup: ArchLinux, CUDA 11.2, cuDNN 8.1, CPU with AVX2 instructions support.

Luckily, the Archlinux community maintains the `tensorflow-opt-cuda` package that perfectly matches the requirements. So, the straightforward solution, if we are interested only in the C & C++ libraries + the headers, is to install this package.

```sh
# pacman -S tensorflow-opt-cuda
```

However, we are interested in **building** TensorFlow from source since we need the building environment to create the custom op (or for customizing the framework itself, it may be useful for creating merge request to TensorFlow), so we use as a reference the [`PKGBUILD`](https://github.com/archlinux/svntogit-community/blob/packages/tensorflow/trunk/PKGBUILD) maintained by [Sven-Hendrik Haase](mailto:svenstaro@gmail.com) to build TensorFlow.

---

The way you need to customize the TensorFlow source code for building it highly depends on your local environment. TensorFlow depends on several third-party libraries and if your local version of these libraries does not match with the one that TensorFlow uses you have to patch the TensorFlow source code or change the library version in your system (**not** recommended since you will likely break other installed software depending on them).

Depending on the reason we are installing TensorFlow from source, the correct branch should be chosen:

- Adding some functionality? Clone the master branch.
- Adding a custom op? Choose the latest stable branch (at the time of writing, `r2.4`).

```bash
git clone git@github.com:tensorflow/tensorflow.git
cd tensorflow
git checkout r2.4
```

In the [Archlinux community repository](https://github.com/archlinux/svntogit-community/tree/packages/tensorflow/trunk), we find the required patches to apply for making the TensorFlow source code compatible with the libraries installed in the system (some patch about h5fs and mkl).

Moreover, all the Python libraries installed on the system might be incompatible with the libraries required by TensorFlow, so we need to remove the explicit dependencies on these fixed versions from the [tools/pip_package/setup.py](https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/tools/pip_package/setup.py) file.

```bash
sed -i -E "s/'([0-9a-z_-]+) .= [0-9].+[0-9]'/'\1'/" tools/pip_package/setup.py
```

Anyway, if you are using a different OS your mileage might vary... a lot. Finding the compatibility problems and how to fix them is a long and boring trial & error process.

### Configuring the building process - hello Bazel

TensorFlow uses [Bazel](https://bazel.build/) as the build tool of choice. Extremely complicated, because extremely configurable, Bazel requires a long set of options for being correctly configured.

TensorFlow comes with a [configure.py](https://github.com/tensorflow/tensorflow/blob/r2.4/configure.py) script, that dynamically reads from the environment or asks the user all the needed info to start the building process. The best way of configuring Bazel is by setting the environment variables before executing the configure.py script. The configure.py script is only (!) 1505 lines, so you can imagine how customizable the building process is.

Luckily again, in the PKGBUILD we can find the minimal set of environment variables to set needed to successfully run the configuration script.

```bash
  # These environment variables influence the behavior of the configure call below.
  export PYTHON_BIN_PATH=/usr/bin/python
  export USE_DEFAULT_PYTHON_LIB_PATH=1
  export TF_NEED_JEMALLOC=1
  export TF_NEED_KAFKA=1
  export TF_NEED_OPENCL_SYCL=0
  export TF_NEED_AWS=1
  export TF_NEED_GCP=1
  export TF_NEED_HDFS=1
  export TF_NEED_S3=1
  export TF_ENABLE_XLA=1
  export TF_NEED_GDR=0
  export TF_NEED_VERBS=0
  export TF_NEED_OPENCL=0
  export TF_NEED_MPI=0
  export TF_NEED_TENSORRT=0
  export TF_NEED_NGRAPH=0
  export TF_NEED_IGNITE=0
  export TF_NEED_ROCM=0
  # See https://github.com/tensorflow/tensorflow/blob/master/third_party/systemlibs/syslibs_configure.bzl
  export TF_SYSTEM_LIBS="boringssl,curl,cython,gif,icu,libjpeg_turbo,lmdb,nasm,pcre,png,pybind11,zlib"
  export TF_SET_ANDROID_WORKSPACE=0
  export TF_DOWNLOAD_CLANG=0
  export TF_NCCL_VERSION=2.8
  export TF_IGNORE_MAX_BAZEL_VERSION=1
  export TF_MKL_ROOT=/opt/intel/mkl
  export NCCL_INSTALL_PATH=/usr
  export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
  export HOST_C_COMPILER=/usr/bin/gcc
  export HOST_CXX_COMPILER=/usr/bin/g++
  export TF_CUDA_CLANG=0  # Clang currently disabled because it's not compatible at the moment.
  export CLANG_CUDA_COMPILER_PATH=/usr/bin/clang
  export TF_CUDA_PATHS=/opt/cuda,/usr/lib,/usr
  export TF_CUDA_VERSION=$(/opt/cuda/bin/nvcc --version | sed -n 's/^.*release \(.*\),.*/\1/p')
  export TF_CUDNN_VERSION=$(sed -n 's/^#define CUDNN_MAJOR\s*\(.*\).*/\1/p' /usr/include/cudnn_version.h)
  export TF_CUDA_COMPUTE_CAPABILITIES=5.2,5.3,6.0,6.1,6.2,7.0,7.2,7.5,8.0,8.6

  export TF_NEED_CUDA=1 # enable cuda
  export CC_OPT_FLAGS="-march=haswell -O3" # AVX2 optimizations
```

Through these variables, we can toggle almost every feature of the framework: the AWS support? The ability to use `s3://` URIs, support for AMD ROCK, and so on.

The most important variables are:

- `TF_NEED_CUDA` - enables CUDA support.
- `CC_OPT_FLAGS` - compiler flags. Aggressive optimizations `-O3` & Intel Haswell architecture (enables AVX2).
- `TF_SYSTEM_LIBS` - contains the list of the libraries we link from our system.
- `TF_CUDA_COMPUTE_CAPABILITIES` - allow us to compile the CUDA kernels only for a subset of the devices. Since this is a build for a specific machine it makes no sense to leave all the items of this list. For example, owning an Nvidia 1080 Ti I can compile only for devices with compute capability 6.1 (see [CUDA: GPUs supported @ Wikipedia](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)).

So after setting these environment variables, we are ready to run the configuration script that generates all the boilerplate required by Bazel.

```
./configure # or directly call the configure.py script
```

The configure script will generate a bunch of Bazel files (like `.tf_configure.bazelrc`) and update others `.bazelrc`. All these files are configuration files used by Bazel.

The root of the TensorFlow repository already contains a `WORKSPACE` file - this instructs Bazel to consider this location root of the building workspace.

Looking inside the WORKSPACE file we can find something like

```bzl
workspace(name = "org_tensorflow")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
```

The first line is just the definition of the workspace (so they give it the name "org_tensorflow"), the second line contains the loading of the `http_archive` function from the "@bazel_tools//tools/build_defs/repo:http.bzl" repository.

This "@bazel_tools" is not a folder inside the TensorFlow repository, but it is an [**undocumented built-in repository**](https://github.com/bazelbuild/bazel/issues/4301), containing some helpful function to download stuff from the internet - it is used to download some pre-trained model that will be shipped with your TensorFlow installation.

TensorFlow requires a specific version of Bazel - another symptom of the Bazel Hell. The approach I recommend is to install [bazelisk](https://github.com/bazelbuild/bazelisk) and let the tool download and use the specified version of Bazel for you (otherwise you have to install Bazel systemwise with the required specific version and this is a pain).

Continue reading the WORKSPACE file we find several `load` statements similar to the one mentioned earlier. These statements just load and execute functions, most of them are inside the `tensorflow/tensorflow.bzl` repository that's another small file (only 2897 lines!) that contains other loads and other function definitions.

Bazel has its own [concepts and terminology](https://docs.bazel.build/versions/master/build-ref.html) - this link is worth a read otherwise all the Bazel files/commands will be very difficult to understand. But a TL;DR is:

- Bazel organizes code in WORKSPACE. A workspace is a directory that contains all the folder and source files you want to compile. A WORKSPACE must be self-contained (you can't refer to header libraries that are outside this workspace -> you can't use your system libraries easily)
- Bazel has its programming language called Starlark - that looks like Python3, but it isn't. There are several limitations and differences respect to Python (reference: [Starlark Language](https://docs.bazel.build/versions/master/skylark/language.html)).
- Starlark is used to define how the software is built - it is both a programming and a configuration language.
- The Bazel code is organized in repositories (the `@` we've seen earlier in the `@bazel_tools` is used to identify the main repository location - that's like the Bazel standard library). The repositories have the `.bzl` extension.
- The code must be organized in **packages**. A package is a directory containing a file named `BUILD` or `BUILD.bazel`.
- The `BUILD` file contains the "target definitions". Targets are files and rules.
- A **rule** specifies the relationship between input and output, and the step to go from input to output. The rules contain the description of every step of the compilation: what to compile, how to compile, the flags to pass, the dependencies between other rules (rules can define compilation target), and what to generate.
- There are different types of build rules. The `*_binary` that builds executable programs (**NOTE**: a .so is generated by these kinds of rules). The `*_test` that are a specialization of  `*_binary` and runs automated tests, and `*_library` that specifies separately-compiled modules.
- Every target has a unique name, identified by its full path using the syntax `//package-name/other-package:target-name`, where `//` identifies the `WORKSPACE` location, and `target-name` a name defined inside the `BUILD` file of `other-package`.

This is the minimal lexical needed to at least have an idea of what's going on when we see a Bazel project.

---

Anyway, after this relatively short digression on Bazel, we can go back to the TensorFlow compilation. We have configured Bazel, and so we might think we are ready to use it. Of course not! We have to know what **targets** build and if there are particular options required to build them for the target setup.

We are interested in:

- Building the TensorFlow C++ libraries (`libtensorflow_cc.so` and `libtensorflow_framework.so`). Target `//tensorflow/libtensorflow_cc.so` and `//tensorflow/libtensorflow_framework.so`.
- Building the TensorFlow C library (useful for creating language bindings or at lest, for having a stable API. The TensorFlow C API is stable and it won't change). Target `//tensorflow/libtensorflow.so`.
- Have the C and C++ headers. Target `//tensorflow:install_headers`.
- Building the Python Wheel. Target `//tensorflow/tools/pip_package:build_pip_package`.

From the PKGBUILD we can find also all the required CLI flags we need to pass to Bazel to build it correctly in Archlinux.

```bash
# Required until https://github.com/tensorflow/tensorflow/issues/39467 is fixed.
export CC=gcc
export CXX=g++

export BAZEL_ARGS="--config=mkl -c opt --copt=-I/usr/include/openssl-1.0 \
                   --host_copt=-I/usr/include/openssl-1.0 --linkopt=-l:libssl.so.1.0.0 \
                   --linkopt=-l:libcrypto.so.1.0.0 --host_linkopt=-l:libssl.so.1.0.0 \
                   --host_linkopt=-l:libcrypto.so.1.0.0"

# Workaround for gcc 10+ warnings related to upb.
# See https://github.com/tensorflow/tensorflow/issues/39467
export BAZEL_ARGS="$BAZEL_ARGS --host_copt=-Wno-stringop-truncation"
```

With the `-c opt` flag we change the [compilation mode to `opt`](https://docs.bazel.build/versions/master/user-manual.html#semantics-options), and with the others {c,link}opts we can specify the typical compiler flags, to add include paths and specifying the linking libraries.

After this last configuration, we are ready to build TensorFlow.

```bash
bazel build ${BAZEL_ARGS[@]} \
  //tensorflow:libtensorflow.so \
  //tensorflow:libtensorflow_cc.so \
  //tensorflow:libtensorflow_framework.so \
  //tensorflow:install_headers \
  //tensorflow/tools/pip_package:build_pip_package
```

About 3 hours later we'll get all our targets built, all of them are in the `bazel-bin` folder. Note: this is a symlink to a folder inside  `~/.cache/bazel`.

The `build_pip_package` target produces an executable file that we must use to generate the Python wheel.

```sh
# Generate the wheel in the /tmp/ folder
bazel-bin/tensorflow/tools/pip_package/build_pip_package --gpu /tmp/
```

The wheel can now be used inside a virtualenv or installed systemwise (not recommended).

All the headers and libraries are also available in the `bazel-bin` folder, you can install it in your system or use it in your executables.

Now that we have the TensorFlow source code ready to use, and we've seen that we're able to compile it from scratch, we can try creating a custom op.

## TensorFlow Custom Ops

The official tutorial [Create an op](https://www.tensorflow.org/guide/create_op) contains some good information about the process required to build a custom op. We'll follow it step-by-step until we reach the point of compiling it using Docker - we'll see what happens if we try to use Docker instead of building it from the local TensorFlow folder and we'll start seeing the first ABI compatibility issues (yay?).

Let's start by reporting the first note of the tutorial

> **Note**: To guarantee that your C++ custom ops are ABI compatible with TensorFlow's official pip packages, please follow the guide at [Custom op repository](https://github.com/tensorflow/custom-op). It has an end-to-end code example, as well as Docker images for building and distributing your custom ops.

We'll be back on ABI compatibility soon.

Anyway, the process of custom op definitions is "straightforward" (when dealing with TensorFlow in C++, simplicity is not the rule, hence the double quotes).

1. Register the operation. The registration is just the description of the operation: name, inputs, outputs, and shape. The register operation is an "abstract concept".
2. Implement the operation (aka **kernel**). Passing from a concept to a "physical" implementation of it. There can be multiple kernels for different input/output types or architectures. Well-known Kernels are the CUDA kernels, which are the implementation of common operations using CUDA (e.g. the convolution operation on NVIDIA GPU uses a CUDA kernel).
3. Want to use it in Python? Create a Python wrapper.
4. Want to use it during a train (via gradient descent)? Write a function to compute gradients for the op.
5. Write the tests!

There's no need to report here the content of the official tutorial, so to implement your first custom op ("ZeroOut"), skip the "Multi-threaded CPU kernels" and "GPU kernels" sections and reach the [Compile the op using Bazel (TensorFlow source installation)](https://www.tensorflow.org/guide/create_op#compile_the_op_using_bazel_tensorflow_source_installation) paragraph.

We can place the `zero_out.cc` file into the folder `tensorflow/core/user_ops/zero/` and also create the `BUILD` file for creating the Bazel package.

The BUILD file content follows

```bzl
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)
```

The `tf_custom_op_library` imported from the `tensorflow.blz` repository is the recommended way for declaring the rule that will produce our `.so` file, anyway this is not the only way, other Bazel rules that can be used to generate shared objects (like `cc_binary` that produces a .so - differently from `cc_library`).

The `name` attribute is the target name, the `srcs` is the list of the sources to compile, and that's it. With a simple Bazel command (we re-pass the same flags used for building TensorFlow in order to be 100% sure that the generated .so is compatible with our compiled TensorFlow version) we can generate the shared library and use it from python easily.

```sh
bazel build ${BAZEL_ARGS[@]} //tensorflow/core/user_ops/zero:zero_out.so
```

**NOTE**: the official tutorial is deprecated and still uses `tf.Session` - the correct way of loading and using the shared object follows

```python
zero = tf.load_op_library("./zero_out.so")
print(zero.zero_out([1,2,3]).numpy()) # [1, 0, 0]
```

it works!

Now we can move on and see if everything works when we add CUDA to the equation (spoiler: it doesn't).

## Custom Ops with GPU support

Following the [Create an op: GPU kernels](https://www.tensorflow.org/guide/create_op#gpu_kernels) paragraph of the official documentation, we end up with 3 files: `kernel_example.cc`, `kernel_example.cu.cc`, `kernel_example.h`.

There is exactly ZERO information on how to compile them, so we have to find this by ourselves.

To compile them using Bazel, we have to create by ourselves a new package in `tensorflow/core/user_ops/example_gpu` and in the BUILD file place

```bzl
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    # kernel_example.cc  kernel_example.cu.cc  kernel_example.h
    name = "kernel_example.so",
    srcs = ["kernel_example.h", "kernel_example.cc"],
    gpu_srcs = ["kernel_example.cu.cc"],
)
```

All the CUDA files must go in the `gpu_srcs` field (this has been found looking into the other TensorFlow's Bazle packages).

If we try to compile it with `bazel build ${BAZEL_ARGS[@]} //tensorflow/core/user_ops/custom_gpu:kernel_example.so` we obtain the first amazing

```error
fatal error: tensorflow/core/user_ops/custom_gpu/example.h: No such file or directory
```

So the tutorial is wrong - let's fix it. The file `kernel_exaple.cu.cc` is wrong - it includes `example.h` but this file doesn't exist. You must change the include to `kernel_example.h`.

Let's try again:

```error
kernel_example.h:13:23: error: 'Eigen' was not declared in this scope
[...]
kernel_example.h:13:42: error: wrong number of template arguments (1, should be 2)
[...]
```

Woah! The dependency on Eigen is not satisfied. After digging in the thousands of BUILD files in the TensorFlow source code I found that the correct way of depending on Eigen (that's in the third-party folder) is by adding the dependency in the `deps` section:

```bzl
deps = ["//third_party/eigen3"]
```

However if we add this line, since we are using the `tf_custom_op_library` rule and not `cc_library` or `cc_binary` we got a new error:

```
Label '//third_party/eigen3:eigen3' is duplicated in the 'deps' attribute of rule 'kernel_example.so
```

Hence `tf_custom_op_library` already includes Eigen, so it's the source code of the example (again) that's wrong. In `kernel_example.h` we have to include

```cpp
#include <unsupported/Eigen/CXX11/Tensor>
```

that's where `Eigen::GpuDevice` is defined.

We can no try again (!) to compile - and we have another error.

```error
1: Compiling tensorflow/core/user_ops/custom_gpu/kernel_example.cu.cc failed: undeclared inclusion(s) in rule '//tensorflow/core/user_ops/custom_gpu:kernel_example_gpu':
this rule is missing dependency declarations for the following files included by 'tensorflow/core/user_ops/custom_gpu/kernel_example.cu.cc':
  'tensorflow/core/user_ops/custom_gpu/kernel_example.h'
```

What does it mean? The `kernel_example.cu.cc` includes `kernel_example.h` but `kernel_example.h` is not in the  `gpu_srcs`, so we have to add it also there. Here's the final BUILD file.

```bzl
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    # kernel_example.cc  kernel_example.cu.cc  kernel_example.h
    name = "kernel_example.so",
    srcs = ["kernel_example.h", "kernel_example.cc"],
    gpu_srcs = ["kernel_example.cu.cc", "kernel_example.h"],
)
```

Success! It builds, and we now have the `kernel_example.so` library ready for being loaded in Python? No, we don't.

The `kernel_example.so` library is just a shared library, without any operation inside. In fact, this is a Kernel (e.g. the implementation of an operation), but there's no `REGISTER_OP` call inside these 3 files, hence these are just implementations of something never defined.

Looking at what the `ExampleFunctor` does in both CPU and GPU code, it looks like this is an implementation of the `input * 2` operation. So to use this operation in Python, we have to register it and re-compile everything. For registering the operation, and make it working with any numeric type, we can add the following lines in `kernel_example.cc`.

```cpp
REGISTER_OP("Example")
    .Attr("T: numbertype")
    .Input("input: T")
    .Output("input_times_two: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```

Re-building the operation, we can now use it in Python in this way:

```python
example_op = tf.load_op_library("./kernel_example.so").example
with tf.device("/gpu:0"):
    tf.assert_equal(example_op([1, 2, 3, 4, 5]), [2, 4, 6, 8, 10])
```

So far, so good (more or less). We've been able to build a CPU & GPU operation correctly, that's perfectly compatible with our TensorFlow installation.

Now that we "master" the Custom Op (in the user-op folders, so I like to call them user ops) concepts let's see what happens if we try to follow the standard path used for building a TensorFlow ops from Google Research.

## Adapting a custom op to a user op: 3D submanifold sparse convolution

[TensorFlow 3D](https://ai.googleblog.com/2021/02/3d-scene-understanding-with-tensorflow.html) is a library recently (February 2021) released by the Google Research team with to bring 3D deep learning capabilities into TensorFlow.

The 3D data captured by sensors often consists of a scene containing interesting objects surrounded by non-interesting parts (background), hence 3D data is inherently sparse. For this reason, it's preferred to avoid traditional convolution operations but uses a particular version of convolution that only focuses on "what matters". This type of convolution is called **submanifold sparse** and is implemented as a [custom op in the TensorFlow 3D repository](https://github.com/google-research/google-research/tree/master/tf3d/ops)).

Let's see what happens then when we follow the process described in the repository, which involves the usage of Docker to avoid the ABI compatibility issues.

We start with the (fixed) script to set up all the required stuff (note that we need the TensorFlow repo to copy the dependencies that are in the `third_party` folder, and also the `custom-op` repo that is a template repository that contains some boilerplate code required to build the operation inside the container.

```bash 
git clone git@github.com:google-research/google-research.git

git clone https://github.com/tensorflow/tensorflow
cd tensorflow && git checkout v2.3.0 && cd ..

git clone https://github.com/tensorflow/custom-op --depth=1

export TF_FOLDER="$(pwd)/tensorflow"
export CUSTOM_OP_FOLDER="$(pwd)/custom-op"

cd google-research/tf3d

mkdir -p tf3d/ops/third_party
cp -a ${TF_FOLDER}/third_party/eigen3 ${TF_FOLDER}/third_party/mkl \
${TF_FOLDER}/third_party/toolchains ${TF_FOLDER}/third_party/BUILD \
${TF_FOLDER}/third_party/eigen.BUILD \
${TF_FOLDER}/third_party/com_google_absl_fix_mac_and_nvcc_build.patch \
${TF_FOLDER}/third_party/com_google_absl.BUILD \
${TF_FOLDER}/third_party/cub.BUILD ${TF_FOLDER}/third_party/repo.bzl \
tf3d/ops/third_party/
cp -a ${CUSTOM_OP_FOLDER}/gpu ${CUSTOM_OP_FOLDER}/tf \
${CUSTOM_OP_FOLDER}/configure.sh
```

Now we can move inside the container, build the operation and have the `.so` ready to use.

```
docker pull tensorflow/tensorflow:2.3.0-custom-op-gpu-ubuntu16
docker run --runtime=nvidia --privileged  -it -v $(pwd)/ops:/working_dir -w /working_dir  tensorflow/tensorflow:2.3.0-custom-op-gpu-ubuntu16

# Inside the container
./configure.sh
bazel run sparse_conv_ops_py_test  --experimental_repo_remote_exec --verbose_failures
cp -a bazel-bin/tensorflow_sparse_conv_ops/_sparse_conv_ops.so tensorflow_sparse_conv_ops/
quit
```

Now in the `tensorflow_sparse_conv_ops` folder we have the `_sparse_conv_ops.so` library ready to use!

Let's see what happens when we try to load it:

```python
tf.load_op_library("./_sparse_conv_ops.so")
```

and we got this (expected) error:

```error
tensorflow.python.framework.errors_impl.NotFoundError: libcudart.so.10.1: cannot open shared object file: No such file or directory1
```

This is expected because, as stated at the beginning of the article, TensorFlow officially supports CUDA 10, while we have installed CUDA 11 on our system.

Anyway, we can try to rebuild the custom op **without** the GPU support to see if it is possible to load the shared library when we don't depend on CUDA.

Inside the container, we just need to define the variable `TF_NEED_CUDA=0` and execute the configure + Bazel build. We end up with a new shared object we can try to load.

```python
tf.load_op_library("./_sparse_conv_ops.so")
```

et voilà

```error
tensorflow.python.framework.errors_impl.NotFoundError: ./_sparse_conv_ops.so: undefined symbol: _ZNK10tensorflow8OpKernel11TraceStringERKNS_15OpKernelContextEb
```

Say hello to the ABI compatibility nightmare.

## The ABI compatibility nightmare

An Application Binary Interface (ABI) is the set of supported **runtime interfaces** provided by a software component for applications to use, differently from the API that is the set of **build-time interfaces**.

A shared object (dynamically linked library) is the most important ABI: the actual linking between an application and a shared object is determined at runtime, therefore if libraries and applications do not use the same common and stable ABI, they cannot work together (they are binarily incompatible, the worst type of incompatibility).

The "undefined symbol" error previously faced, happens when an application loads a shared object, and the shared object needs some function from other libraries / the current runtime - but it can't find them.

It's pretty unclear what symbol our `_sparse_conv_ops.so` requires because it's mangled (the C++ identifier must be translated to a C-compatible identifier because linkers only support C identifiers for symbol names), but we can demangle it using `c++filt`.

```bash
echo "_ZNK10tensorflow8OpKernel11TraceStringERKNS_15OpKernelContextEb" | c++filt

# tensorflow::OpKernel::TraceString(tensorflow::OpKernelContext const&, bool) const
```

It's pretty clear that `_sparse_conv_ops.so` requires this function from some of the TensorFlow C++ libraries (`libtensorflow_framework.so` and `libtensorflow_cc.so`). Let's see if the symbol is present. When `nm` we can examine binary files and display their symbol table and other meta information.

```bash
nm -D /usr/lib/libtensorflow_cc.so  |c++filt  | grep "tensorflow::OpKernel::TraceString"

# 0000000000fb8f20 T tensorflow::OpKernel::TraceString[abi:cxx11](tensorflow::OpKernelContext const&, bool) const
```

Can you see the difference?

We are looking for `tensorflow::OpKernel::TraceString(tensorflow::OpKernelContext const&, bool) const` but we have `tensorflow::OpKernel::TraceString[abi:cxx11](tensorflow::OpKernelContext const&, bool) const`. The **[abi:cxx11]** tag indicates that our `libtensorflow_cc.so` has been compiled with the **new** ABI that's not binary compatible (that's why they have different names) with the old ABI that's been used inside the container to generate the shared object.

What to do now? We only have 1 option: rebuild tensorflow, again, defining the `_GLIBCXX_USE_CXX11_ABI=0` macro through Bazel (as indicated in the [Troubleshooting](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html) section of the libstd++ documentation.

Alright, let's rebuild TensorFlow by asking the compiler to generate the old ABI symbols.

```bash
bazel build ${BAZEL_ARGS[@]} --copt=-D_GLIBCXX_USE_CXX11_ABI=0 \
  //tensorflow:libtensorflow.so \
  //tensorflow:libtensorflow_cc.so \
  //tensorflow:libtensorflow_framework.so \
  //tensorflow:install_headers \
  //tensorflow/tools/pip_package:build_pip_package
```

After the usual 3 hours, we end up with a new TensorFlow runtime (we have to re-install in the system the library and install the new Python wheel).

Another try of loading the sparse op without the GPU support

```python
sparse_conv_lib = tf.load_op_library("./_sparse_conv_ops.so")
```

It works!! 🎉

We can now use the sparse convolution operation on our custom TensorFlow setup, but only with CPU support...

Since inside the container there's CUDA 10, we can't use it - even the nightly container, still uses CUDA 10 so it's incompatible with our setup. The only option we have is trying to migrate the operation from the "custom op" to a "user op". However, since the article is becoming too long, this is going to be covered in the next part.

## Conclusion

Having a custom TensorFlow setup built from source allows complete customization of the framework: we can enable/disable features, enable device-specific optimizations, tailor the framework on our hardware. However, if our running environment is not compatible with the officially supported setup (Ubuntu, CUDA 10), doing it is not straightforward. Moreover, customizing the framework via Custom Ops is a really nice feature that allows us to create shared objects usable from Python in a relatively easy way. The ABI compatibility should always be taken into account when creating shared objects (especially on different environments like the containers) and the dependencies on other runtime libraries (like CUDA) can cause other headaches.

In the next article, we'll see how to migrate from custom op to user op, or better, we'll try :)

For any feedback or comment, please use the Disqus form below - thanks!
