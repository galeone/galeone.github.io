---
layout: post
title:  "Integrating third-party libraries as Unreal Engine plugins: solving the ABI compatibility issues on Linux when the source code is available"
date:   2023-06-18 8:00:00
summary: "In this article, we will discuss the challenges and potential issues that may arise during the integration process of a third-party library when the source code is available. It will provide guidance on how to handle the compilation and linking of the third-party library, manage dependencies, and resolve compatibility issues. We'll realize a plugin for redis plus plus as a real use case scenario, and we'll see how tough can it be to correctly compile the library for Unreal Engine - we'll solve every problem step by step."
authors:
    - pgaleone
    - ChatGPT
---

Some time ago, I wrote an article titled [Integrating third-party libraries as Unreal Engine plugins: ABI compatibility and Linux toolchain](/2022/05/27/unreal-engine-third-party-linux-abi-compatibility/) where I explained how to correctly design an Unreal Engine plugin and introduced the ABI compatibility problems that may arise when integrating a pre-built library (no source code available).

To briefly recap, when integrating a third-party library there are 2 possible scenarios:

1. The third-party library comes as a pre-built binary.
2. The third-party library source code is available and we have to compile it.

The aforementioned article covered the first point and highlighted that we can do **nothing** if the source code is not available and the library is compiled with libstdc++ - that's not ABI compatible with the C++ standard library used by Unreal on Linux: libc++.

In this article instead, we will discuss the challenges and potential issues that may arise during the integration process of a third-party library when the source code is available. It will provide guidance on how to handle the compilation of the library source code, manage dependencies, and resolve compatibility issues.

## The source code available scenario

Having the source code is great, but even in this case the compilation of the third-party library to work within Unreal Engine on Linux may be a painful process (and perhaps impossible!). There are several things to take into account in the scenario:

- Do we know the build system the third-party library uses?
- Is the third-party library well-written and explicitly mentions all the dependencies?
- Are the conflicts between the library and Unreal Engine?

The second and the third point are tightly bounded because conflicts are likely to happen. After all, the engine is HUGE and inside it contains tons of libraries, each of them at a specific version.

What if our third-party library depends on a specific version of another library (e.g. something common like [libjpeg](https://libjpeg.sourceforge.net/)) while the engine depends on the very same library but at a completely different version? (e.g. Version 1.0.0 with a certain public API vs version 5.0.0 with a completely different public API)

Here, there's no standard solution and two possible paths are possible

1. We change the engine source code. If we are lucky, the engine depends on the library only on some files, so it should be "quite easy" to upgrade the engine to use the new API of the dependency.
2. If, instead, the engine depends heavily on this third-party library (e.g. SDL)? Then touching the engine would be a really tough option and we should instead downgrade (I suppose the engine uses an old version of the library) the dependency of our third-party library, and update its source code instead.

Both options require rewriting a lot of source code, but this is the best way to proceed if it's impossible to have in the same binary two libraries at different versions (and this happens more than you think).

The first point - the build system used by the third-party library - can be another great pain point. If the project uses Bazel, you need to understand how Bazel works, if it uses CMake too, if it uses Meson too, and so on ...

### The third-party library with source code available

Let's deep dive into a real scenario: creating the Unreal Engine plugin for [redis-plus-plus](https://github.com/sewenew/redis-plus-plus).

Let's start this journey in the easiest possible way:

1. Compile the library as described in the [README](https://github.com/sewenew/redis-plus-plus#install-redis-plus-plus) 
1. Create the Unreal Engine plugin structure, create the `ThirdParty` folder and the `RedisPlusPlus` **external** module inside it (as described [in the previous article](/2022/05/27/unreal-engine-third-party-linux-abi-compatibility/#creating-an-unreal-plugin)
1. Define and Implement the Public interface of the Plugin (we'll implement only a single function)
1. Try to use the Plugin and see what happens.


#### Compiling Redis Plus Plus

Straight from the README: we only need to install the only **explicit** dependency of the library that's [hiredis](https://github.com/redis/hiredis). However, the `CMakeLists.txt` contains several `find_library` statements, and thus there are a lot of other libraries that are needed to correctly build Redis Plus Plus. Anyway, let's ignore this for now, and let's build the library.

```bash
git clone https://github.com/sewenew/redis-plus-plus.git
cd redis-plus-plus
mkdir build
cd build
cmake ..
make
```

Now, instead of installing the library somewhere in the system, we use the `DISTDIR` parameter of make to change the destination of the `make install` command.

First, we create a folder and then we install the headers and the binaries there.

```bash
mkdir /tmp/rd
make DESTDIR=/tmp/rd install
```

At this point we have `/tmp/rd/usr/local/lib/` containing the libraries and `/tmp/rd/usr/local/include/` containing the library headers.

We are now ready to create the plugin structure and copy the shared objects and the headers to the plugins' destination folder. The *complete* plugin creation part is not explained here since it has been already covered in the [previous article](/2022/05/27/unreal-engine-third-party-linux-abi-compatibility/#creating-an-unreal-plugin) - anyway the external module build file and its content are detailed, as well as the public interface of the plugin.

**NOTE**: the explicit dependency on hiredis must be satisfied even if hiredis is already installed in the system. In fact, UBT is only aware of the libraries available **in the engine** and not in the system (and this is a good thing, otherwise we may end up developing unreal games that are working fine on our machines are but not easy to redistribute because depending on system libraries that the users should install by themselves).

Let's also build hiredis in the very same way we built Redis Plus Plus. The clean solution would have been to create a dedicated External module in the plugin (next to the external module `RedisPlusPlus`) and make `RedisPlusPlus` depend on `HiRedis`. But this is left to the reader. For now we just put the headers of hiredis inside the same `Public` folder of `RedisPlusPlus` and the libraries in the same library.

```
git@github.com:redis/hiredis.git
cd hiredis
mkdir build
cd build
cmake ..
make
mkdir /tmp/hr
make DESTDIR=/tmp/hr install
```

So far so good, we can now move on and create the plugin.

## Create the Unreal Engine plugin structure

Below is just presented the plugin structure together with the location of the libraries inside the RedisPlusPlus plugin.

`sw` is the header path of redis-plus-plus, `hiredis` is the folder where the hirediss headers have been placed. Both the compiled libraries have been placed inside the `x64` folder.

```tree
RedisPlusPlus
└── Source
    ├── RedisPlusPlus
    │   ├── Private
    │   └── Public
    └── ThirdParty
        └── RedisPlusPlusLibrary
            ├── include
            │   ├── hiredis
            │   └── sw
            │       └── redis++
            └── linux
                └── x64
```

The UBT it's not happy to work with symlinks, so I removed the symlinks and only obtained `libredis++.so` and `libhiredis.so` that are the only 2 files in the `x64` folder. For the sake of completeness, here's the `RedisPlusPlusLibrary.Build.cs` file content

```csharp
public class RedisPlusPlusLibrary : ModuleRules {
  public RedisPlusPlusLibrary(ReadOnlyTargetRules Target) : base(Target) {
    Type = ModuleType.External;

    PublicIncludePaths.Add(Path.Combine(ModuleDirectory, "include"));

    if (Target.Platform == UnrealTargetPlatform.Linux) {
      string libname = "libredis++.so";
      string path = Path.Combine(ModuleDirectory, "linux", "x64", libname);
      PublicAdditionalLibraries.Add(path);
      RuntimeDependencies.Add(path);

      libname = "libhiredis.so";
      path = Path.Combine(ModuleDirectory, "linux", "x64", libname);
      PublicAdditionalLibraries.Add(path);
      RuntimeDependencies.Add(path);
    }
  }
}
```

Now that the external module has been defined and the libraries and the headers have been placed in the correct locations, we can define and implement the `RedisPlusPlus` public interface.

### Define and Implement the Public interface of the Plugin

As stressed in the [previous article](/2022/05/27/unreal-engine-third-party-linux-abi-compatibility/#creating-an-unreal-plugin), there must be a complete segregation of the third-party library and the plugin's public interface. That's why, for declaring the private member of type `sw::redis::Redis` we mustn't include `"sw/redis++/redis.h"` that's where the type is defined, but we need to forward declare it.


Here's the public interface (`RedisPlusPlus.h`)

```cpp
// Forward declare private type in the public header
namespace sw::redis {
class Redis;
}

class REDISPLUSPLUS_API FRedisPlusPlus : public IModuleInterface {
  public:
  // There's no need to override IModuleInterface
  // StartupModule & ShutdownModule methods.
  
  void Connect(const FString &Host, int32 Port = 6379);

  FString Ping(TOptional<FString> Message) const;

  private:
  TUniquePtr<sw::redis::Redis> _instance{nullptr};
};
```

And here's the private part, where we can include the third-party headers and use them. All the considerations made in the [previous article](/2022/05/27/unreal-engine-third-party-linux-abi-compatibility/) about the string conversions still hold of course.

```cpp
void FRedisPlusPlus::Connect(const FString &Host, const int32 Port) {
  sw::redis::ConnectionOptions Opts;
  Opts.host = TCHAR_TO_UTF8(*Host);
  Opts.port = Port;
  _instance = MakeUnique<sw::redis::Redis>(Opts);
  check(_instance);
}

FString FRedisPlusPlus::Ping(TOptional<FString> Message) const {
  check(_instance);
  std::string Reply{};
  if (Message.IsSet()) {
    Reply = _instance->echo(TCHAR_TO_UTF8(*Message.GetValue()));
  } else {
    Reply = _instance->ping();
  }
  return UTF8_TO_TCHAR(Reply.c_str());
}
```

Let's go compile it.

### Missing symbols

Of course (that's the point of this article) there are problems. The code compiles successfully but during the linking phases 2 errors arise

```
undefined symbol: sw::redis::Redis::echo(std::__1::basic_string_view<char, std::__1::char_traits<char> > const&)
undefined symbol: sw::redis::Redis::ping()
```

This topic has already been covered in the [previous article](/2022/05/27/unreal-engine-third-party-linux-abi-compatibility/), so here we'll just use `nm` and see what symbols are inside the `libredis++.so`. Let's focus only on the `ping` method

```nm
nm -D libredis++.so | grep ping |c++filt
000000000004dc10 W sw::redis::cmd::ping(sw::redis::Connection&)
000000000004d090 W sw::redis::cmd::ping(sw::redis::Connection&, std::basic_string_view<char, std::char_traits<char> > const&)
0000000000042bd0 T sw::redis::Redis::ping[abi:cxx11](std::basic_string_view<char, std::char_traits<char> > const&)
0000000000044a70 T sw::redis::Redis::ping[abi:cxx11]()
```

The library contains the symbol `sw::redis::Redis::ping[abi:cxx11]()` while the linker is looking for `sw::redis::Redis::ping()`. Moreover, the `echo` method accepts a `string_view` as input, and we know (see previous article) that the strings can be binary incompatible when working with different standard libraries (the thing that we are doing right now).

So the problem is that the library has been built using `libstdc++` while Unreal uses `libc++`, so when Unreal Engine's compiler generates the method signature from the third-party library header, it generates it to be compatible with `libc++`, and thus the generated symbol is different and can't be found inside the library.

## Changing sysroot

On Linux, Unreal Engine ships a complete development environment with a toolchain and a whole sysroot to use. The sysroot is a folder containing the whole standard structure of the Linux filesystem.

Here's the location and its content (the path is relative to the engine's root).

```bash
ls Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v20_clang-13.0.1-centos7/x86_64-unknown-linux-gnu/

bin  include  lib  lib64  libexec  share  usr
```

Using a sysroot, we can tell the build system (CMake in this case) to not use our system libraries but work as if the root of the filesystem is the sysroot passed.

Thus, we now need to re-compile our third-party library but specify the sysroot. From the [CMake documentation](https://cmake.org/cmake/help/git-master/variable/CMAKE_SYSROOT.html) it looks like the only way to set a sysroot is to change the `CMakeLists.txt` file and add the line

```cmake
set(CMAKE_SYSROOT path_of_the_sysroot)
# in my case
# set(CMAKE_SYSROOT $ENGINE/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v20_clang-13.0.1-centos7/x86_64-unknown-linux-gnu/)
```

So, after deleting the folder `build` in the `redis-plus-plus` repository, and deleted also the destination directory previously used (`/tmp/rd`) we can change the sysroot to the one provided by the engine, and try to compile the project.

```bash
cd build
cmake ..
make
```

we end up with several compilation errors all of the looking like

```
redis-plus-plus/src/sw/redis++/errors.h:22:10: fatal error: hiredis/hiredis.h: No such file or directory
   22 | #include <hiredis/hiredis.h>
```

This happens because the sysroot is an isolated environment and thus CMake can't find the `hiredis` library. To fix this, we can manually move the hiredis headers and libraries inside the sysroot.

**IMPORTANT**: this is possible without recompiling the `hiredis` library only because `hiredis` is a C library, and thus there are no ABI compatibility issues. If we were depending on other C++ libraries, we shall have to backtrack and re-compile every single libraries and their dependency changing the sysroot, and installing them inside the sysroot itself (to make them available for the other compilations).

```bash
# Copy the hiredis headers from the plugin to the sysroot in the include folder
cp -r RedisPlusPlus/Source/ThirdParty/RedisPlusPlusLibrary/include/hiredis/ \
      $ENGINE/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v20_clang-13.0.1-centos7/x86_64-unknown-linux-gnu/include/

# Copy the hiredis libraries from the plugin to the sysroot in the lib64 folder
cp -r RedisPlusPlus/Source/ThirdParty/RedisPlusPlusLibrary/linux/x64/libhiredis.so \
      $ENGINE/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v20_clang-13.0.1-centos7/x86_64-unknown-linux-gnu/lib64/
```

After deleting the `build` folder, and running `cmake ..` once again, this error disappeared but a new one appears:

```
In file included from /usr/include/c++/13.1.1/mutex:45,
                 from /home/pgaleone/redis-plus-plus/src/sw/redis++/connection_pool.h:22,
                 from /home/pgaleone/redis-plus-plus/src/sw/redis++/connection_pool.cpp:17:
/usr/include/c++/13.1.1/bits/std_mutex.h: In member function ‘void std::__condvar::wait_until(std::mutex&, clockid_t, timespec&)’:
/usr/include/c++/13.1.1/bits/std_mutex.h:185:7: error: ‘pthread_cond_clockwait’ was not declared in this scope; did you mean ‘pthread_cond_wait’?
  185 |       pthread_cond_clockwait(&_M_cond, __m.native_handle(), __clock,

```

There's something strange going on here: the error comes from the standard library implementation of `mutex`. While compiling `connection_pool.cpp` and including `connection_pool.h`, but as it can be read, the problem comes from `/usr/include/c++/13.1.1/mutex` that's a location NOT relative to the sysroot, so CMake is searching outside of the sysroot and this is a problem. We need to configure CMake in a better way:

- Specify the C compiler
- Specify the C++ compiler
- Force to not search for programs outside of the sysroot

All these options are pretty standard while cross-compiling - and even if we are not really cross-compiling, since the host target is the very same as the destination target, we must use them to prevent this kind of leakage.


```cmake
set(CMAKE_SYSROOT      $ENGINE/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v20_clang-13.0.1-centos7/x86_64-unknown-linux-gnu/)
set(CMAKE_C_COMPILER   $ENGINE/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v20_clang-13.0.1-centos7/x86_64-unknown-linux-gnu/bin/clang)
set(CMAKE_CXX_COMPILER $ENGINE/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v20_clang-13.0.1-centos7/x86_64-unknown-linux-gnu/bin/clang++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
```

However, once again, something really weird happens when compiling:

```
/home/pgaleone/redis-plus-plus/src/sw/redis++/cxx17/sw/redis++/cxx_utils.h:20:10: fatal error: 'string_view' file not found
#include <string_view>
         ^~~~~~~~~~~~~
```

How is it possible to not have the `string_view` header when Unreal compiles successfully using the C++17 standard?

Looking inside the sysroot, it's pretty clear that there's no `string_view` header! Instead, we can find it in a very unusual location:

```
./Engine/Source/ThirdParty/Unix/LibCxx/include/c++/v1/string_view
```

So, Unreal Engine decided to do some very unusual thing: it ships a sysroot that's completely unrelated to the standard library used while compiling Unreal Engine projects! Instead, we have the `Engine/Source/ThirdParty/Unix/LibCxx/` location that's **NOT** a sysroot, but it contains only the libc++ headers and binaries to use while compiling and linking.

How can we set CMake to use the bundled clang (that's in the sysroot) and instead ignore the libraries and the header of the sysroot and use the one provided in the `LibCxx` folder?

## The definitive compilation environment for third-party libraries

The C and C++ compilers are correct. The sysroot is correct too (because the binaries are there, like the compilers, the linker, the assembler, ...), but we need to **disable** both the standard includes and the default libraries, and set the one available in the `LibCxx` folder.

Cmake comes with the support for doing all these things (and all the decent build system allows doing it). The correct way is to not change the original `CMakeLists.txt` but instead create a new CMake file (called `UE4ToolChain.cmake`) that contains all the directives to set.

```cmake
set(ENGINE "/home/pgaleone/ue/engine/")

set(CMAKE_SYSROOT "${ENGINE}/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v20_clang-13.0.1-centos7/x86_64-unknown-linux-gnu/")
set(CMAKE_C_COMPILER "${ENGINE}/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v20_clang-13.0.1-centos7/x86_64-unknown-linux-gnu/bin/clang")
set(CMAKE_CXX_COMPILER "${ENGINE}/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v20_clang-13.0.1-centos7/x86_64-unknown-linux-gnu/bin/clang++")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nostdinc++ -I${ENGINE}/Engine/Source/ThirdParty/Unix/LibCxx/include/ -I${ENGINE}/Engine/Source/ThirdParty/Unix/LibCxx/include/c++/v1/")
```

Of course, you need to change the `$ENGINE` variable to your engine location.


We can now create once again the `build` folder and invoke CMake specifying the toolchain to use:

```bash
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../UEToolchain.cmake ..
make
```

Voilà! Now `string_view` exists, but (hurray?) a new problem appears:

```
[ 97%] Built target redis++
In file included from /home/pgaleone/redis-plus-plus/test/src/sw/redis++/test_main.cpp:34:
In file included from /home/pgaleone/redis-plus-plus/test/src/sw/redis++/redlock_test.h:47:
/home/pgaleone/redis-plus-plus/test/src/sw/redis++/redlock_test.hpp:39:10: fatal error: 'openssl/rc4.h' file not found
#include <openssl/rc4.h>
         ^~~~~~~~~~~~~~~
```

As anticipated, there are hidden dependencies inside the project, one of these dependencies is on OpenSSL that's not available in the sysroot. How can we fix it?

### Moving the engine libraries inside the sysroot

One possibility is the find the location of the headers and libraries of the OpenSSL third-party library available in the engine and move them to the sysroot, where our compiler can find them.

Searching inside the engine codebase the `rc4.h` file we find

```
./Engine/Source/ThirdParty/OpenSSL/1.1.1n/include/Unix/x86_64-unknown-linux-gnu/openssl/rc4.h
```

We can than copy the `openssl` folder inside the `include` folder of the sysroot, and doing the same for the libraries that are available in

```
./Engine/Source/ThirdParty/OpenSSL/1.1.1n/lib/Unix/x86_64-unknown-linux-gnu/
```

This is *safe* and we don't need to re-compile anything, since it comes directly from the engine and thus it's 100% compatible (same C++ version, same standard library, ...).

```bash
# headers
cp -r ./Engine/Source/ThirdParty/OpenSSL/1.1.1n/include/Unix/x86_64-unknown-linux-gnu/openssl/ \
      $ENGINE/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v20_clang-13.0.1-centos7/x86_64-unknown-linux-gnu/include/
# libraries
cp -r ./Engine/Source/ThirdParty/OpenSSL/1.1.1n/lib/Unix/x86_64-unknown-linux-gnu/* \
      $ENGINE/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v20_clang-13.0.1-centos7/x86_64-unknown-linux-gnu/lib64/
```

Once again `rm -rf build; mkdir build; cd build; cmake -DCMAKE_TOOLCHAIN_FILE=../UEToolchain.cmake ..` and it works<sup>\*</sup>!

<sup>\*</sup>*Partially, the tests are throwing errors because CMake when it does `add_subdirectory` does NOT propagate the toolchain, so if we are interested in the tests we need to manually pass the toolchain file to the `CMakeLists.txt` in the `test` folder.*

We can now copy to the `x64` folder of the plugin the brand new `libredis++.so` library and see what happens now.

```
0>Link (lld) libUnrealEditor-RedisPlusPlus-Linux-DebugGame.so [ Time 0.04 s ]
0>
0>Total time in Parallel executor: 0.34 seconds
0>Total execution time: 1.58 seconds
Build succeeded at 6:03:22 PM
```

Here we go - it really works!

## Conclusion

In conclusion, integrating third-party libraries as Unreal Engine plugins can be a complex task, especially when dealing with ABI compatibility and the Linux toolchain. In this article, we discussed the challenges and potential issues that arise when integrating a third-party library with available source code.

When working with the source code of a third-party library, the compilation process can still be challenging. It is important to consider factors such as the build system used by the library, explicit mention of dependencies, and potential conflicts with the Unreal Engine itself.

Understanding the build system used by the third-party library, such as Bazel, CMake, or Meson, is crucial for successful integration.

To overcome the challenges related to system libraries, the concept of sysroot was introduced. Unreal Engine provides a complete development environment with a sysroot, which allows specifying an isolated environment for compilation. By setting the sysroot to the engine's provided path, the compilation process can use the appropriate system libraries and resolve header file dependencies.

However, adjusting the sysroot alone is not sufficient for resolving all compilation errors. Especially because Unreal decided to ship a sysroot with a standard library that's not the standard library used inside the Engine itself! For this reason, we introduced a CMake toolchain file and make the build system point to the files in the `LibCxx` Unreal module, containing the headers and the libraries of libc++. Some additional manual steps, such as moving the required headers and libraries into the sysroot, may be necessary to ensure a successful compilation.

In summary, integrating third-party libraries with available source code into Unreal Engine requires careful consideration of build systems, explicit dependencies, and potential conflicts. Managing ABI compatibility and addressing linking issues are essential steps in achieving successful integration. With proper understanding and troubleshooting, it is possible to incorporate third-party libraries into Unreal Engine plugins on Linux while maintaining compatibility and functionality.
