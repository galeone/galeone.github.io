---
layout: post
title: "Fixing the codesigning and notarization issues of Unreal Engine (5.3+) projects"
date: 2024-07-04 08:00:00
categories: unrealengine macos
summary: ""
authors:
    - pgaleone
---

Starting from Unreal Engine 5.3, Epic Games added the support for the so called [modern Xcode workflow](https://dev.epicgames.com/documentation/en-us/unreal-engine/using-modern-xcode-in-unreal-engine-5.3-and-newer). This workflow allows the Unreal Build Tool (UBT) to be more consistent with the standard Xcode app projects, and to be compliant with the Apple requirements for distributing applications... In theory! 😅 In practice this workflow is flawed: both the code signing and the framework supports are not correctly implemented, making the creation of working apps and their distribution impossible.

In this article we'll go trough the problems faced during the packaging, codesigning, and notarization of an Unreal Engine application that includes:

- The CrashReporter: so an additional binary inside our .app.
- The browser widget (so the engine CEF plugin - that is now treated as a macOS/iOS framework).
- A third party library with pre-built libraries ([URedis](https://github.com/Galeontz/URedis)).
- Engine third party libraries (like Intel tbb - that's automatically added in the packaged application).

The problems presented are inside the Engine itself, and we try to fix them from the outside avoiding (when possible!) to modify the engine.

## Creating distribution-signed code for macOS

After creating an Unreal Engine application we, of course, want to distribute it to the world. On macOS we can decide to publish our app to the Mac App Store or to distribute it directly to our users (e.g. making it available for download on our website).

In both cases, we need to create distribution-signed code which is code that's being compiled, signed, packaged, signed once again, and either submitted to the Mac App Store or to the notary service.

One of the promised of the modern Xcode workflow is to simplify the developers' life integrating the steps of compilation, sign and packaging inside the UBT. It will be the UBT itself to invoke Xcode and to perform the signing steps.

### Configure the modern Xcode workflow for code signing

The configuration can be done using the editor itself or - as I recommend - editing the configuration files manually (so we don't have to open the editor that's a slow operation).

The file to configure is the `DefaultEngine.ini` configuration file, in the local `Config` directory. As every unreal developer knows these files are able to overwrite the settings already defined at engine level (settings that we can copy to use as a starting point). `[/Script/MacTargetPlatform.XcodeProjectSettings]` is the section to configure.

```toml
[/Script/MacTargetPlatform.XcodeProjectSettings]
bUseModernXcode=true
bUseAutomaticCodeSigning=true
bMacSignToRunLocally=false
CodeSigningTeam=<TEAM IDENTIFIER>
BundleIdentifier=<APPLICATION IDENTIFIER>
TemplateMacPlist=(FilePath="/Game/Build/Mac/Resources/Info.plist")
PremadeMacEntitlements=(FilePath="/Game/Build/Mac/Resources/entitlements.plist")
ShippingSpecificMacEntitlements=(FilePath="/Game/Build/Mac/Resources/entitlements.plist")
```

- `bUseModernXcode=true`: enables the modern Xcode workflow.
- `bUseAutomaticCodeSigning=true`: enables the code signing using the information specified below.
- `bMacSignToRunLocally=false`: must be set to false to be able to firm with a valid developer ID, and not with the empty id "-".
- `CodeSigningTeam`, and `BundleIdentifier` are the unique identifiers of the Apple Developers (given by apple) and of the application under the same team ID, respectively.
- `TemplateMacPlist` is the soft-path (relative to the `Game` directory, that's where the `.uproject` file is) of the *information property list* (`Info.plist`). This file will be placed inside the application bundle and it contains a set of metadata (in a key-value fashion) used by both the application itself, the operating system, or by the system frameworks (e.g. to facilitate the launch of apps).
- `PremadeMacEntitlements` and `ShippingSpecificMacEntitlements` are the soft-paths of another `.plist` file that can contain other metadata. The file contains the list of rights/privilege that the application requires to run (as we'll see in the [TODO LINK TO SECTION], we need to grant certain privileges to our app to be able to execute the web browser widget without a crash).

To get the `<APPLICATION IDENTIFIER>` and `<TEAM IDENTIFIER>` I redirect the reader to the Unreal Engine documentation about this topic: [Provisioning Profiles and Signing Certificates](https://dev.epicgames.com/documentation/en-us/unreal-engine/setting-up-ios-tvos-and-ipados-provisioning-profiles-and-signing-certificates-for-unreal-engine-projects). Please also note that you must login on Xcode using your Apple Developer Account. This is mandatory to correctly start the signing process using Xcode (invoked by the UBT).

## The application & the distribution workflow

The `Config/DefaultEngine.ini` file is part of very trivial unreal engine application that, as mentioned in the first paragraph of the article, contains a couple of third-party plugins (`URedis` and `WebBrowserWidget`). The full project can be found here: [galeone/ue-bundle-project](https://github.com:galeone/ue-bundle-project).

Briefly, the application contains an empty world with just an user widget, containing a Web Browser Widget pointing to this website. The world also contains an `ARedis` actor that `OnBeginPlay` it connects to the local Redis server, sets a value, fetch this value, and prints in the scene the retrieved value. That's all.

In the article only the relevant parts for packaging, codesigning, and notarization are shown.

We are interested in redistributing this application, and for doing it we need to:

1. Create the app - this is the standard packaging process of the application
2. Sign the app and its content.
3. Create a signed `.pkg`.
4. Send to the notarization server the `.pkg` to have the final verification from Apple that the application has been correctly signed, and contains the correct security features and the users can safely download the application and execute it (without being asked to trust the author of the app because Apple trusted this developer and the application - so offering to the users a better installation experience).

Moreover, we want to insert in the bundle the crash report client provided by the engine. This is a separate application that will be bundled inside the `.app` automatically called when the application crashes, useful to collect the crash reports.

## Packaging & integrated code signing

The packaging process allows us to get a (correctly?) signed application. Using [ue4cli](https://github.com/adamrehn/ue4cli) we can easily invoke the UBT to create a shipping package with the crash report client inside. If you are following these steps using [galeone/ue-bundle-project](https://github.com:galeone/ue-bundle-project) you need to edit the `Config/DefaultEngine.ini`, `Build/Mac/Resources/Info.plist`, and `Build/Mac/Resources/entitlements.plist` replacing the `REPLACE_WITH_TEAM_ID` and `REPLACE_WITH_BUNDLE_ID` with the appropriate values.

```sh
LC_ALL="C" ue4 package Shipping -CrashReportClient
```

This will take a while. Once completed, the `BundleProject-Mac-Shipping.app` has been created inside the `dist/Mac` folder. Using `codesign` we can verify if the content has been signed (note: not correctly signed as requested by Apple for distribution!).


```sh
codesign --verify --verbose dist/Mac/BundleProject-Mac-Shipping.app
```
```text
dist/Mac/BundleProject-Mac-Shipping.app: valid on disk
dist/Mac/BundleProject-Mac-Shipping.app: satisfies its Designated requirement
```

Everything looks OK - but unfortunately, if we try to open this brand new app, it just crashes without giving us any clue.

Unreal Engine from version 5.3 onward changed the flags required to create a valid package (only on macOS apparently). In fact, we can see the app content to only contain the main executable, but there are no dylibs! We expect to have at least the libraries of the URedis plugin.

```tree
```
