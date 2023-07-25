---
layout: post
title:  "Git & Unreal Engine: semantic versioning, plugins, and git submodules"
date:   2023-07-23 5:00:00
summary: ""
authors:
    - pgaleone
---

Developing games with Unreal Engine is a fantastic experience. You can go from zero to a complete game in just a few weeks. The key to fast development is the usage of the plugins: instead of reinventing the wheel, you can use these libraries (the plugins) that implement what you need (if you are lucky) or that expose functionalities that you can use as a starting point.

However, unreal engine plugins are mainly distributed through the [Unreal Engine Marketplace](https://www.unrealengine.com/marketplace/en-US/store) which allows us to download or buy the plugin only knowing the Unreal Engine version compatible and the supported operating systems. The marketplace doesn't allow the developers to know if a new release of the plugin has been released; thus, no [semantic versioning](https://semver.org/) is enforced or even supported.

Plugins developers, thus, usually try to release a new version of the plugin only when a new engine version is released. This is really unfortunate since unreal engine releases are not frequent while the development of a plugin is something that requires frequent iterations and, thus, frequent releases. Even worse, if they release a new plugin release (for the same engine version) with some breaking changes in the public API there's no way to know about it or to see the differences between the previous and the current release.

Version control systems, like diversion, git, and perforce, are the tools to use for both the development and the release of an unreal engine project or plugin. These tools, allow you to correctly apply semantic versioning, and when we do it everyone benefits from it. We can release a plugin when we want, and we can make it clear when we introduce breaking changes or bug fixes.

In this article, we are going to show how to develop a plugin following the rules of semantic versioning, how to use git while developing the plugin, and to conclude we'll show how to integrate the plugin in an unreal engine project as a git submodule.

<!--
https://news.ycombinator.com/item?id=31703803
-->
