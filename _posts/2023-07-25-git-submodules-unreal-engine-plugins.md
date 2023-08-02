---
layout: post
title:  "Git & Unreal Engine: semantic versioning, plugins, and git submodules"
date:   2023-07-23 5:00:00
summary: ""
authors:
    - pgaleone
---

Developing games with Unreal Engine is a fantastic experience. You can go from zero to a complete game in just a few weeks. The key to fast development is the usage of the plugins: instead of reinventing the wheel, you can use these libraries (the plugins) that implement what you need (if you are lucky) or that expose functionalities that you can use as a starting point.

However, Unreal Engine plugins are mainly distributed through the [Unreal Engine Marketplace](https://www.unrealengine.com/marketplace/en-US/store) which allows us to download or buy the plugin only knowing the Unreal Engine version compatible and the supported operating systems. The marketplace doesn't allow the developers to know if a new release of the plugin has been released; thus, no [semantic versioning](https://semver.org/) is enforced or even supported.

Plugins developers, thus, usually try to release a new version of the plugin only when a new engine version is released. This is unfortunate since Unreal Engine releases are not frequent while the development of a plugin is something that requires frequent iterations and, thus, frequent releases. Even worse, if they release a new plugin release (for the same engine version) with some breaking changes in the public API there's no way to know about it or to see the differences between the previous and the current release.

Version control systems, like diversion, git, and perforce, are the tools to use for both the development and the release of an Unreal Engine project or plugin. These tools, allow you to correctly apply semantic versioning, and when we do it everyone benefits from it. We can release a plugin when we want, and we can make it clear when we introduce breaking changes or bug fixes.

In this article, we are going to show how to develop a plugin following the rules of semantic versioning, how to use git while developing the plugin, and to conclude we'll show how to integrate the plugin into an Unreal Engine project as a git submodule.

## Unreal Engine Plugins

Plugins are the Unreal Engine's way of implementing libraries. Every plugin has a well-defined structure that clearly separates private and public parts. In the `Public` folder, there are only the public headers of the library, while in the `Private` folder, there are all the `.cpp` files, as well as the headers that we want to keep private.

Correctly separating the private parts and the public parts is the first step to correctly structure a plugin. In fact, the public part of the plugin will be the unique access point to the plugin functionality. Being libraries, the most important part is a correct definition of the Public API (Application Programming Interface).

The creation and the anatomy of an Unreal Engine plugin are perfectly covered in the [official documentation](https://docs.unrealengine.com/5.2/en-US/plugins-in-unreal-engine/) and for this reason, it won't be covered in this article.

I want to stress once again how important is the definition of the public API of the plugin, for several reasons:

1. The public API is the sole entry point for the plugin. The users will only use public classes and invoke public functions.
1. As a plugin developer, you are defining a contract between your service (the plugin) and your clients (the plugin's users). You want to offer stability to your clients, and you don't want to break the contract (changing in a certain way the public API - as we'll see later).
1. If you really need to break the contract, you have to correctly tell your users that something really important is changed and they need to upgrade.

As mentioned in the article introduction, using the Unreal Engine marketplace is a good way for distributing (sell) your plugin, but not ideal to redistribute the plugin with frequent releases because you can only guarantee that your plugin is compatible with certain engine versions, but there's no support for versioning the plugin itself.

Anyway, why one should bother with this versioning thing? The reason is in the contract defined by the public API, and versioning is the correct way to make the contract explicit.


## Semantic Versioning

[Semantic Versioning](https://semver.org/spec/v2.0.0.html) is the de-facto standard used in the software development industry to version libraries. The introduction of the semver specification contains a description, quoted below, that describes the principles of semantic versioning

> Once you identify your public API, you communicate changes to it with specific increments to your version number. Consider a version format of X.Y.Z (Major.Minor.Patch). Bug fixes not affecting the API increment the patch version, backward compatible API additions/changes increment the minor version, and backward incompatible API changes increment the major version.
>
> We call this system “Semantic Versioning.” Under this scheme, version numbers and the way they change convey meaning about the underlying code and what has been modified from one version to the next.

So, the first thing is to **identify the public API**. During the early stage of development, this is quite impossible; frequent changes are the norm and no stability can be guaranteed. That's why the specification itself, states that every version 0.x.y must be considered unstable and can change in every moment. In this way, the developer can work on the API and the users know that the contract can change at any moment.

Once the public API has been defined we, as plugin developers, go into bugfixes and enhancement mode. When fixing a bug, if this fix is backward compatible, we only increment the PATCH counter. If we are adding new functionalities in a backward-compatible manner, we increment the MINOR counter (and set the PATCH counter to 0). When, instead, we change the contract, and so there's the introduction of a non-backward compatible change, we must release a new MAJOR version.

To clarify these 3 scenarios, we will now develop a simple plugin for mathematical operations and we'll show how to use `git tag` to implement semantic versioning.

## Unreal Engine plugin and Git for managing versions

`git` is a distributed version control system widely used, perhaps the most widely used. In the game development industry, other tools like Perforce are more complex to use and manage. That's why git is getting more traction day by day for its simplicity even among game developers.

Let's create our math plugin. The creation process can be breakdown into the following steps:

1. Create an empty folder (`mathplugin` and initialize the git repository
   ```bash
   mkdir mathplugin
   cd mathplugin
   git init
   ```
2. Create the empty structure, following the [official documentation](https://docs.unrealengine.com/5.2/en-US/plugins-in-unreal-engine/). We end up with this structure:
   ```bash
    MathPlugin
    ├── MathPlugin.uplugin
    └── Source
        └── MathPlugin
            ├── MathPlugin.Build.cs
            ├── Private
            │   └── MathPlugin.cpp
            └── Public
                └── MathPlugin.h

   ```
3. Create the public API for our plugin. `MathPlugin.h` contains the declaration of the `FMathPluginModule`. We can add to this file other declarations and start defining the public API of our plugin. We want to support some basic operations like sum, subtraction, division, and multiplication for FVectors.

   ```cpp
   // Public/MathPlugin.h
   namespace MathPlugin
   {
       // Add two vectors
       FVector AddVectors(FVector a, FVector b);

       // Subtract two vectors
       FVector SubtractVectors(FVector a, FVector b);

       // Multiply two vectors
       FVector MultiplyVectors(FVector a, FVector b);

       // Divide two vectors
       FVector DivideVectors(FVector a, FVector b);
   };
   ```
4. Implement the private part. As plugin developers, we are the only ones that can see how things are implemented. Our users will only use the public API defined above.

   ```cpp
   // Private/MathPlugin.cpp
   namespace MathPlugin {
   FVector AddVectors(FVector a, FVector b) {
     for (int i = 0; i < 3; i++) {
       a[i] += b[i];
     }
     return a;
   }

   FVector SubtractVectors(FVector a, FVector b) {
     for (int i = 0; i < 3; i++) {
       a[i] -= b[i];
     }
     return a;
   }

   FVector MultiplyVectors(FVector a, FVector b) {
     for (int i = 0; i < 3; i++) {
       a[i] *= b[i];
     }
     return a;
   }

   FVector DivideVectors(FVector a, FVector b) {
     for (int i = 0; i < 3; i++) {
       a[i] /= b[i];
     }
     return a;
   }
   }; // namespace MathPlugin
   ```
5. Create the first release! We suppose that our API is already stable, the library does what it has been designed to. So we can go straight to release `1.0.0`.

   ```bash
   git add . # add everything inside the folder
   git commit -m 'Basic functionalities added'
   git tag v1.0.0 -m 'Release version 1.0.0'
   ```

   If we have added a remote (that's the remote location of this git repository, usually hosted on services like Github/Bitbucket/Gitlab), we can `git push` to the remote and allow our users to clone and checkout to the precise version `1.0.0` or our library. The default remote is called `origin` and we can push all the changes and the tags with a simple command

   ```
   git push origin --tags
   ```

So far so good.

Our users can now add the plugin to their Unreal Engine project (that uses `git` as version control) as a git submodule (that's a git repository inside a git repository) specifying that they precisely want to use version `1.0.0` of our plugin. We will see how they will do it in the  [Plugins as Git submodules](#plugins-as-git-submodules) section.


### Creating a patch release

Our first release can be improved. The implementation of all the methods can be changed from the `for` loop based, to the direct usage of the `FVector` operators. This type of change is completely safe, it's a bugfix/enhancement and it doesn't change the signature of the public methods.

```cpp
FVector AddVectors(FVector a, FVector b) {
  return a+b;
}

FVector SubtractVectors(FVector a, FVector b) {
  return a-b;
}

FVector MultiplyVectors(FVector a, FVector b) {
  return a*b;
}

FVector DivideVectors(FVector a, FVector b) {
  return a/b;
}
```

This change perfectly matches the rule given by the semantic versioning specification about when to increment the PATCH counter. So we can now move on, and create a new version.

```bash
git add .
git commit -m 'Internal change'
git tag v1.0.1 -m 'Release version 1.0.1'
git push origin --tags
```

Our clients can safely upgrade from `1.0.0` to `1.0.1` and everything will work in the very same way.

### Creating a minor release

We want to expand the functionalities of our library, giving the user the possibility of computing the power of every component of an `FVector`. So, as usual, we define the public API by adding the new function signature in the `MathPlugin.h` public header:

```cpp
// Public/MathPlugin.h

namespace MathPlugin {
    // Exponentiate a vector
	FVector ExponentiateVector(FVector a, double exponent);
};
```

In the private part, we add the implementation

```cpp
// Private/MathPlugin.cpp

#include <cmath>

FVector ExponentiateVector(FVector a, double exponent) {
  for (int i = 0; i < 3; i++) {
    a[i] = std::pow(a[i], exponent);
  }
  return a;
}
```

This kind of change matches the rule given by the semantic versioning specification about when to increment the MINOR counter. In fact, this is the addition of new functionality in a backward-compatible manner.

We can release version `1.1.0` and our users can upgrade safely without worrying about breaking changes.

```bash
git add .
git commit -m 'Added ExponentiateVector'
git tag v1.1.0 -m 'Release version 1.1.0'
git push origin --tags
```

Even after upgrading the plugins, our clients' code will work as usual even though a new functionality is now available.


### Creating a major release

Being redundant, we decide to **remove from our library** the basic functionalities to add, subtract, multiply, and divide FVectors; after all these functionalities are already implemented by Unreal and there's no need to wrap them.

Removing something from the public API is a huge breaking change, and thus we must create a new major release to address this change.

After removing the functions we can release the 2.0.0 version.

```bash
git commit -m 'Removing redundant functions'
git tag v2.0.0 -m 'Release version 2.0.0'
git push origin --tags
```

What happens if our client code upgrades? The clients will notice the changes in the major version and they now have two options:

1. Decide to **do not upgrade** and stick with the 1.x.y version they are using
1. Upgrade the plugin to the latest version, and fix their code.

The latter scenario is the only case in which our clients must spend time upgrading their codebase because they upgraded the plugin.

So far, we've seen what is like to use semantic versioning when developing a plugin. Let's move to the other side, and let's see how git and git submodules can be used pretty well to add plugins to your Unreal Engine project and why this is a good choice.


## Plugins as Git submodules

Similarly to what has been done with the plugin creation, we now create a blank Unreal Engine project `MyProject` with the minimal set of files that define an Unreal project:

```
MyProject
├── MyProject.uproject
├── Plugins
└── Source
    ├── MyProject
    │   ├── MyProject.Build.cs
    │   ├── MyProject.cpp
    │   ├── MyProjectGameModeBase.cpp
    │   ├── MyProjectGameModeBase.h
    │   └── MyProject.h
    ├── MyProjectEditor.Target.cs
    └── MyProject.Target.cs
```

We initialize `MyProject` as a git repository and we commit the first version.


```bash
cd MyProject
git init
git add .
git commit -m 'Empty project'
```

We are now interested in adding the `MathPlugin` previously created to `MyProject`.

`git` supports nesting git repositories using the so-called git submodules. `git submodule` is the command that allows us to manage these nested repositories.

```bash
git submodule add -b tags/v1.0.0 <remote_url> Plugins/MathPlugin
```

With this command, we are adding the math plugin previously created in the `Plugins/MathPlugin` folder of their project.

- `<remote_url>` is the location of the git repository on a git server like GitHub.
- `-b tags/v1.0.0` is the flag to use to specify that we are interested **exactly** to version 1.0.0 of the plugin

So far so good. We can see all the git submodules added to our project using the `git submodule` command:

```bash
3cbbbc3d814d69e1b61e7a4bb3000920e33f3c29 Plugins/MathPlugin (v1.0.0)
```

In this example, we only have a single submodule inside the `Plugins/MathPlugin` folder, checked out a specific commit tagged with version `v1.0.0`.

Changing the directory to the plugin itself (`cd Plugins/MathPlugin`) we can use the standard git commands to work in this submodule in the very same way we could work in every other git repository.

### Upgrading plugins using submodules

From time to time, we might be interested in seeing if there are updates in the plugins we are using. In our example, we can just move into the plugin repository and update the remote.

```bash
git fetch
```

This command fetches from the remote counterpart (by default, the `origin` remote) all the branches and tags. After running it, we can use `git tag -l` to list all the available tags

```
v1.0.0
v1.0.1
v1.1.0
v2.0.0
```

These are the 3 new versions available. We can without any worry switch to version `v1.0.1` or `v1.1.0` since semantic versioning guarantees us that no braking changes have been added.

Switching to a new version is as easy as typing

```bash
git checkout v1.1.0
```

Using this tag system, it's also possible to see that a new major version has been released (2.0.0), and thus if we want to switch to the latest version with breaking changes inside, we can once again `git checkout`, but this time we must be careful and understand how those breaking changes will affect our codebase.

### Versioning of marketplace's plugin

The workflow I suggest following is to create a private git repository, and to respect the intellectual property of the plugin creator, commit the source code into it. Use the tagging strategy to give meaningful tags to the plugins you downloaded.

Periodically, download the plugin and overwrite its content. Use `git diff` to check if something changed. Of course, you should only focus your effort on the changes in the `Public` folder of the plugin. So you can just run `git diff Public`.

Depending on the type of changes apply to semantic versioning strategies described above to perform a correct versioning of the plugin, even if the plugin's creator is not doing it.

It's a manual process but it's perhaps the best way to have a correct understanding of the plugin you're using. In this way, you keep track of the changes and understand the effort you need to put into updating your codebase (that's using the public interface of the plugin): if nothing changed in the public API, the effort is zero. If there are only additions without modification, the effort is zero once again. If there are changes in the public API you're forced to upgrade your code.

## Conclusion

One of the keys to achieving fast development is leveraging Unreal Engine plugins. These plugins serve as libraries, providing ready-to-use functionalities and allowing developers to avoid reinventing the wheel.

However, there are challenges when it comes to managing and distributing Unreal Engine plugins. The primary distribution platform is the Unreal Engine Marketplace, which lacks support for semantic versioning. As a result, developers often release new plugin versions only when a new engine version is available, which can be limiting, especially considering the need for frequent iterations during plugin development.

To address these issues, developers can turn to version control systems such as Git, Perforce, or Diversion. Git, in particular, is widely used and preferred for its simplicity, even among game developers. By using Git and adhering to semantic versioning principles, developers can release plugins with more frequent updates, clearly communicate changes to users, and provide stability through well-defined public APIs.

In conclusion, developing Unreal Engine plugins with a focus on semantic versioning and leveraging the capabilities of Git for version control can lead to smoother collaboration, more stable plugin releases, and ultimately a more enjoyable game development experience for both developers and users alike.
