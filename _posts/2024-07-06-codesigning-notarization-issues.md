---
layout: post
title: "Fixing the code signing and notarization issues of Unreal Engine (5.3+) projects"
date: 2024-07-06 08:00:00
categories: unrealengine macos
summary: "Starting from Unreal Engine 5.3, Epic Games added support for the so-called modern Xcode workflow. This workflow allows the Unreal Build Tool (UBT) to be more consistent with the standard Xcode app projects, and to be compliant with the Apple requirements for distributing applications... In theory! 😅 In practice this workflow is flawed: both the code signing and the framework supports are not correctly implemented, making the creation of working apps and their distribution impossible. In this article, we'll go through the problems faced during the packaging, code signing, and notarization of an Unreal Engine application on macOS and end up with the step-by-step process to solve them all."
authors:
    - pgaleone
---

Starting from Unreal Engine 5.3, Epic Games added support for the so-called [modern Xcode workflow](https://dev.epicgames.com/documentation/en-us/unreal-engine/using-modern-xcode-in-unreal-engine-5.3-and-newer). This workflow allows the Unreal Build Tool (UBT) to be more consistent with the standard Xcode app projects, and to be compliant with the Apple requirements for distributing applications... In theory! 😅 In practice this workflow is flawed: both the code signing and the framework supports are not correctly implemented, making the creation of working apps and their distribution impossible.

In this article, we'll go through the problems faced during the packaging, code signing, and notarization of an Unreal Engine application that includes:

- The CrashReporter: so an additional binary inside our .app.
- The browser widget (so the engine CEF plugin is now treated as a macOS/iOS framework).
- A third-party library with pre-built libraries ([URedis](https://github.com/Galeontz/URedis)).
- Engine third-party libraries (like Intel TBB - that is automatically added to the packaged application).

The problems presented are inside the Engine itself, and we try to fix them from the outside avoiding (when possible!) modifying the engine.

The article describes all the attempts performed to reach the goal. If you are interested in the steps, you can jump to the [TL;DR](#tldr) section.

## Creating distribution-signed code for macOS

After creating an Unreal Engine application we, of course, want to distribute it to the world. On macOS, we can decide to publish our app to the Mac App Store or to distribute it directly to our users (e.g. making it available for download on our website).

In both cases, we need to create distribution-signed code which is code that is being compiled, signed, packaged, signed once again, and either submitted to the Mac App Store or to the notary service.

One of the promises of the modern Xcode workflow is to simplify the developers' lives by integrating the steps of compilation, code signing, and packaging inside the UBT. It will be the UBT itself to invoke Xcode and to perform the signing steps.

### Configure the modern Xcode workflow for code signing

The configuration can be done using the editor itself or - as I recommend - editing the configuration files manually (so we don't have to open the editor which is a slow operation).

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
- `CodeSigningTeam` and `BundleIdentifier` are the unique identifiers of the Apple Developers (given by Apple) and of the application under the same team ID, respectively.
- `TemplateMacPlist` is the path (relative to the `Game` directory, that is where the `.uproject` file is) of the *information property list* (`Info.plist`). This file will be placed inside the application bundle and it contains a set of metadata (in a key-value fashion) used by both the application itself, the operating system, or by the system frameworks (e.g. To facilitate the launch of apps).
- `PremadeMacEntitlements` and `ShippingSpecificMacEntitlements` are the paths of another `.plist` file that can contain other metadata. The file contains the list of rights/privileges that the application requires to run (as we'll see in [The Info.plist and the entitlements](#the-infoplist-and-the-entitlements) section, we need to grant certain privileges to our app to be able to execute the web browser widget without a crash). The `PremadeMacEntitlements` contains the meta-data applied to every build configuration. The `ShippingSpecificMacEntitlements` can be used to apply shipping-specific metadata, if required.

To get the `<APPLICATION IDENTIFIER>` and `<TEAM IDENTIFIER>` I redirect the reader to the Unreal Engine documentation about this topic: [Provisioning Profiles and Signing Certificates](https://dev.epicgames.com/documentation/en-us/unreal-engine/setting-up-ios-tvos-and-ipados-provisioning-profiles-and-signing-certificates-for-unreal-engine-projects). Please also note that you must log in on Xcode using your Apple Developer Account. This is mandatory to correctly start the signing process using Xcode (invoked by the UBT).

## The application & the distribution workflow

The `Config/DefaultEngine.ini` file is part of a very trivial unreal engine application that, as mentioned in the first paragraph of the article, contains a couple of third-party plugins (`URedis` and `WebBrowserWidget`). The full project can be found here: [galeone/ue-bundle-project](https://github.com:galeone/ue-bundle-project).

Briefly, the application contains an empty world with just a user widget, containing a Web Browser Widget pointing to this website. The world also contains an `ARedis` actor that `OnBeginPlay` it connects to the local Redis server, sets a value, fetches this value, and prints in the scene the retrieved value. That's all.

In the article, only the relevant parts for packaging, code signing, and notarization are shown.

We are interested in redistributing this application, and for doing it we need to:

1. Create the app - this is the standard packaging process of the application.
2. Sign the app and its content.
3. Create a signed `.pkg`.
4. Send to the notarization server the `.pkg` to have the final verification from Apple that the application has been correctly signed, and contains the correct security features and the users can safely download the application and execute it (without being asked to trust the author of the app because Apple trusted this developer and the application - so offering to the users a better installation experience).

Moreover, we want to insert in the bundle the crash report client provided by the engine. This is a separate application that will be bundled inside the `.app` automatically called when the application crashes, useful for collecting the crash reports.

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

Everything looks OK - but unfortunately, if we try to open this brand-new app, it just crashes without giving us any clue.

Unreal Engine from version 5.3 onward changed the flags required to create a valid package (only on macOS apparently). In fact, we can see the app content to only contain the main executable, but there are no dylibs! We expect to have at least the libraries of the URedis plugin. Moreover, there's no reference to CEF (required by the Web Browser widget) nor a reference to the CrashReportClient.app!

```sh
tree -a dist/Mac/BundleProject-Mac-Shipping.app
```
```text
dist/Mac/BundleProject-Mac-Shipping.app
└── Contents
    ├── Info.plist
    ├── MacOS
    │   └── BundleProject-Mac-Shipping
    ├── PkgInfo
    ├── Resources
    │   ├── AppIcon.icns
    │   ├── Assets.car
    │   ├── LaunchScreen.storyboardc
    │   │   ├── 01J-lp-oVM-view-Ze5-6b-2t3.nib
    │   │   ├── Info.plist
    │   │   └── LaunchScreen.nib
    │   └── UEMetadata
    │       └── PrivacyInfo.xcprivacy
    └── _CodeSignature
        └── CodeResources
```

By copying the invocation of the UBT while creating the package from the editor, we find out that right now is required to explicit the `-package` flag.

```sh
LC_ALL="C" ue4 package Shipping -package -CrashReportClient
```

Now, also the name of the application changed from a developer-friendly name to a customer-friendly name (`BundleProject.app` without additional technical information).

If we look inside the bundle we can see almost all the missing parts.

```sh
tree -a dist/Mac/BundleProject.app
```
```text
dist/Mac/BundleProject.app
└── Contents
    ├── Frameworks
    │   └── Chromium Embedded Framework.framework
    │       ├── Chromium Embedded Framework
    │       ├── Libraries
    │       │   ├── libEGL.dylib
    │       │   ├── libGLESv2.dylib
    │       │   ├── libswiftshader_libEGL.dylib
    │       │   ├── libswiftshader_libGLESv2.dylib
    │       │   ├── libvk_swiftshader.dylib
    │       │   └── vk_swiftshader_icd.json
    │       ├── Resources
    │       │   ├── Info.plist
    │       │   ├── icudtl.dat
    │       │   ├── snapshot_blob.bin
    │       │   ├── v8_context_snapshot.arm64.bin
    │       └── _CodeSignature
    │           └── CodeResources
    ├── Info.plist
    ├── MacOS
    ├── Resources
    └── UE
        ├── BundleProject
        │   ├── Binaries
        │   │   └── Mac
        │   ├── BundleProject.uproject
        │   ├── Config
        │   ├── Content
        │   └── Plugins
        │       └── URedis
        │           ├── Source
        │           │   └── ThirdParty
        │           │       └── URedisLibrary
        │           │           └── mac
        │           │               └── arm64
        │           │                   ├── libhiredis.1.1.0.dylib
        │           │                   └── libredis++.1.dylib
        │           └── URedis.uplugin
        ├── Engine
        │   ├── Binaries
        │   │   ├── Mac
        │   │   │   ├── CrashReportClient.app
        │   │   │   └── EpicWebHelper
        │   │   └── ThirdParty
        │   │       ├── Apple
        │   │       ├── Intel
        │   │       │   └── TBB
        │   │       │       └── Mac
        │   │       │           ├── libtbb.dylib
        │   │       │           └── libtbbmalloc.dylib
        │   │       ├── Ogg
        │   │       └── Vorbis
        │   ├── Config
        │   ├── Content
        │   ├── Extras
        │   ├── Plugins
        │   ├── Programs
        │   │   └── CrashReportClient
        │   └── Shaders
        └── UECommandLine.txt
```

Note: the tree output has been post-processed to remove a lot of files that are not useful for the goal of this article.

So far so good? It looks like all the libraries are there, as well as the CrashReporterClient.app.

We can try to execute the application... It crashes. Once again 😭

### Chromium Embedded Framework as a macOS framework

By creating a `Development` package, we can get the application logs and use them to understand what's going on.

```text
[ERROR:icu_util.cc(178)] icudtl.dat not found in bundle
[ERROR:icu_util.cc(242)] Invalid file descriptor to ICU data received.
```

This is strange, since `icudtl.dat` is in the bundle at the path `Contents/Frameworks/Chromium Embedded Framework.framework/Resource`.

Digging into the engine source code looking for the correct CEF location, we can see in `MacPlatform.Automation.cs` this line.

```cs
public override void ProcessArchivedProject(ProjectParams Params, DeploymentContext SC)
{
	// nothing to do with modern
	if (AppleExports.UseModernXcode(Params.RawProjectPath))
	{
        return;
	}
```

This early return prevents the execution of the method `FixupFrameworks`. The first line of this method mentions `Engine/Binaries/ThirdParty/CEF3/Mac` as the target directory for CEF inside the bundle.

However, we don't want to disable the modernized XCode framework, so we have 2 options:

1. Manually move the `Chromium Embedded Framework.framework` folder inside `Engine/Binaries/ThirdParty/CEF3/Mac`. This is a valid option but it requires to codesign the package once again, since changing the content of the `.app` invalidates its signature.
2. Prevent the UBT from creating the Framework folder, and let the UBT copy the framework in the old (correct) location.

This second option requires to modify the build file of CEF `CEF.build.cs`:

```cs
if (Target.LinkType == TargetLinkType.Modular || !AppleExports.UseModernXcode(Target.ProjectFile))
{
	// Add contents of framework directory as runtime dependencies
	foreach (string FilePath in Directory.EnumerateFiles(FrameworkLocation.FullName, "*", SearchOption.AllDirectories))
	{
		RuntimeDependencies.Add(FilePath);
	}
}
// for modern
else
{
	FileReference ZipFile = new FileReference(FrameworkLocation.FullName + ".zip");
	// this is relative to module dir
	string FrameworkPath = ZipFile.MakeRelativeTo(new DirectoryReference(ModuleDirectory));

	PublicAdditionalFrameworks.Add(
		new Framework("Chromium Embedded Framework", FrameworkPath, Framework.FrameworkMode.Copy, null)
		);
}
```

When using modern Xcode we enter in the `else` branch that copies the frameworks in the Framework directory. Instead, we want to always enter in the if branch, thus the code has to be modified accordingly.

After applying this change, we can finally execute the application. Should it work, right? 😅

### The Info.plist and the entitlements

Of course, it doesn't work yet!

```
[FATAL:mach_port_rendezvous.cc(142)] Check failed: kr == KERN_SUCCESS. bootstrap_check_in org.chromium.ContentShell.framework.MachPortRendezvousServer.32575: Permission denied (1100)
```

We have a permission issue. Our application for some reason (that we are going to discover soon) doesn't have permission to execute something. Even trying with root privileges doesn't fix the issue. So? Here's where the entitlements file comes into play.

After hours of trial and error and various research on the [CEF forum](https://magpcss.org/ceforum/viewtopic.php?f=6&t=16215&start=10), I understood that this permission-denied issue is only a matter of correctly setting the Bundle ID in the entitlements.

So, the entitlements that we are going to use **must** have the section `com.apple.application-identifier` set with the same value used in the `DefaultEngine.ini` file while setting the `BundleIdentifier`key. The entitlements file, thus, should contain at least this content (of course replacing the `<APPLICATION IDENTIFIER>` with the correct Bundle ID).

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
       <key>com.apple.application-identifier</key>
       <string><APPLICATION IDENTIFIER></string>
</dict>
</plist>
```

So, the UBT doesn't do this for us although it has all the information needed (Epic Games this is for you).

Moreover, to be 100% consistent (and I'm sorry but I don't have a lot of trust in how Epic Games does this stuff) I also prefer to be explicit and define our `Info.plist` file without hoping for the UBT to correctly generate one for us.

You can see the complete Info.plist in the [repository](https://github.com/galeone/ue-bundle-project/blob/main/Build/Mac/Resources/Info.plist). The important part is to also set there the Bundle ID:

```xml
	<key>CFBundleIdentifier</key>
	<string>REPLACE_WITH_BUNDLE_ID</string>
```

Now, after the first configuration of the entitlements and of the Info.plist, we can package once again our application and try to execute it 🤞

```sh
LC_ALL="C" ue4 package Development -package -CrashReporter
# invoked this way to see the logs, and not invoked using open dist/Mac/BundleProject.app
./dist/Mac/BundleProject.app/Contents/MacOS/BundleProject
```

It works 🙌

## The notarization process

We have a working application. Now we want to package it in a `.pkg` (or `.zip` or `.dmg` it's the same) and ship it to our clients. The first step is the creation of the product and the sign of the bundle with the correct packaging certificate. The certificate to use for the code signing of the product is the one that starts with `Developer ID Installer:`. In the following sections, we assume that the environment variable `$installer_cert` contains that certificate.

`productbuild` is the tool to use for going from a `.app` to a `.pkg`

```sh
productbuild --component dist/Mac/BundleProject.app /Applications \
             --sign "$installer_cert" \
             --timestamp \
             --identifier REPLACE_WITH_BUNDLE_ID \
             BundleProject.pkg
```

As usual, the Bundle ID should be replaced with the correct identifier. After a few seconds, the product is ready and we have our `BundleProject.pkg` ready to be shipped to our clients. Right? Well, no. It depends on the result of the notarization process.

The [Notarizing macOS software before distribution](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution/) page is very clear about what notarization is and why it is required.

> Notarize your macOS software to give users more confidence that the Developer ID-signed software you distribute has been checked by Apple for malicious components. Notarization of macOS software is not App Review. The Apple notary service is an automated system that scans your software for malicious content, checks for code-signing issues, and returns the results to you quickly. If there are no issues, the notary service generates a ticket for you to staple to your software; the notary service also publishes that ticket online where Gatekeeper can find it.

So, let's notarize our product! The notarization process happens on the Apple servers, so we need our developer's credentials. In the following we assume the environment variables `$APPDEV_PASSWORD`, `$APPDEV_ID`, and `$APPDEV_TEAMID` to contain the password, user ID and team ID respectively.

```sh
xcrun notarytool submit BundleProject.pkg \
      --password $APPDEV_PASSWORD --apple-id $APPDEV_ID --team-id $APPDEV_TEAMID \
      --wait --force --verbose --output-format plist
```

The time for executing this command may vary according to the internet speed and how big the `.pkg` is. After a bit of time, here's what we got.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>id</key>
	<string>76801d98-0ce6-4bd1-9fd5-c444f848ac4e</string>
	<key>message</key>
	<string>Processing complete</string>
	<key>status</key>
	<string>Invalid</string>
</dict>
</plist>
```

Quite generic. But of course "Invalid" 😭

Something went wrong and we have no idea about what precisely. Luckily, the `notarytool` can be also used to fetch detailed logs. We need the UUID of our notarization process, which can be found by looking at the output of the `notarytool submit` command previously invoked. It can be seen that every GET request is performed to a URL that ends with a UUID.

```txt
Preparing GET request to URL: https://appstoreconnect.apple.com/notary/v2/submissions/OUR-UNIQUE-UUID-IS-HERE
```

Using this ID, we can use the `notarytool log` and try to understand what's going on.

```sh
xcrun notarytool log OUR-UNIQUE-UUID-IS-HERE \
           --password $APPDEV_PASSWORD --apple-id $APPDEV_ID --team-id $APPDEV_TEAMID
```

The output is a (big) JSON response, with several errors message like:

```json
{
  "severity": "error",
  "code": null,
  "path": "BundleProject.pkg/YOUR.BUNDLE.ID.pkg Contents/Payload/Applications/BundleProject.app/Contents/MacOS/BundleProject",
  "message": "The binary is not signed with a valid Developer ID certificate.",
  "docUrl": "https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution/resolving_common_notarization_issues#3087721",
  "architecture": "arm64"
},
{
  "severity": "error",
  "code": null,
  "path": "BundleProject.pkg/YOUR.BUNDLE.ID.pkg Contents/Payload/Applications/BundleProject.app/Contents/MacOS/BundleProject",
  "message": "The signature does not include a secure timestamp.",
  "docUrl": "https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution/resolving_common_notarization_issues#3087733",
  "architecture": "arm64"
},
{
  "severity": "error",
  "code": null,
  "path": "BundleProject.pkg/YOUR.BUNDLE.ID.pkg Contents/Payload/Applications/BundleProject.app/Contents/MacOS/BundleProject",
  "message": "The executable does not have the hardened runtime enabled.",
  "docUrl": "https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution/resolving_common_notarization_issues#3087724",
  "architecture": "arm64"
},
{
  "severity": "error",
  "code": null,
  "path": "BundleProject.pkg/YOUR.BUNDLE.ID.pkg Contents/Payload/Applications/BundleProject.app/Contents/MacOS/BundleProject",
  "message": "The executable requests the com.apple.security.get-task-allow entitlement.",
  "docUrl": "https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution/resolving_common_notarization_issues#3087731",
  "architecture": "arm64"
},
{
  "severity": "error",
  "code": null,
  "path": "BundleProject.pkg/YOUR.BUNDLE.ID.pkg Contents/Payload/Applications/BundleProject.app/Contents/UE/BundleProject/Plugins/URedis/Source/ThirdParty/URedisLibrary/mac/arm64/libredis++.1.dylib",
  "message": "The binary is not signed with a valid Developer ID certificate.",
  "docUrl": "https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution/resolving_common_notarization_issues#3087721",
  "architecture": "arm64"
}
```

These 4 errors are repeated **for all the libraries and binaries in the bundle**. Every library of the CEF framework, the executable EpicWebHelper, the CrashReportClient.app, and so on.

Those errors are very strange since UBT signed for us the content of the `.app` and the app itself. The only issue that looks easy to solve is the "The executable requests the com.apple.security.get-task-allow entitlement".

After several attempts, I ended up with this entitlements file that satisfies pretty much all the permissions needed for releasing the software and also to be able to debug it.
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
       <key>com.apple.application-identifier</key>
       <string><APPLICATION IDENTIFIER></string>
       <key>com.apple.security.cs.allow-dyld-environment-variables</key>
       <true/>
       <key>com.apple.security.cs.allow-jit</key>
       <true/>
       <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
       <true/>
       <key>com.apple.security.cs.disable-executable-page-protection</key>
       <true/>
       <key>com.apple.security.cs.disable-library-validation</key>
       <true/>
</dict>
</plist>
```

However, updating the entitlements file doesn't solve all the other issues.

## Is the Unreal Engine code signing not enough?

Yes. If you want to be able to redistribute a notarized application, you cannot use the code signing embedded in the unreal editor for 3 reasons:

- There's no way to enable the hardened runtime and add a secure timestamp to the signed files.
- The file `Info.plist` is not correctly generated from the information provided in the `DefaultEngine.ini`. The bundle ID (which is very important) is not updated automatically.
- If you are using the Web Browser Plugin (CEF) - well, it's broken.

So, once reached this stage, we have to manually codesing the `.app` and all its content, being sure to specify all the flags required to enable the hardened runtime and to add the secure timestamp.

Moreover, there's another thing to **carefully** note: the engine provides libraries that have no executable bit set. This is a problem for the notarization problem (I learned this after hours and hours spent attempting to notarize a correctly signed application).

Thus the first step is to ensure that all the `.dylib` provided by the engine and placed inside our `.app` are executable.

```sh
find dist/Mac/BundleProject.app/Contents/UE/Engine/ -name '*dylib' -exec chmod +x {} \;
```

The second step is to codesign all the executable files inside the applications. For this code signing step, we export another environment variable named `$application_cert` that must contain a valid developer ID certificate. The one that starts with `Developer ID Application:`.

```sh
# find all the executable files in the bundle and exclude .sh and .bat files (that for some reason unreal places there, but are not needed)
# replace the new lines with the null byte, so we can support the code signing of file with spaces in the path/name
to_sign=$(find dist/Mac/BundleProject.app/ -type f -perm +111 -print | grep -vE "\.(sh|bat)$" | \tr '\n' '\0')

# code sign all the executables with the application certificate, adding the hardened runtime flag and the secure timestamp
echo -n $to_sign | xargs -t -0 codesign -f -vvv -s "$application_cert" --entitlements Build/Mac/Resources/entitlements.plist --options runtime --timestamp
```

Last but not least, remember to codesign the app itself.

```sh
codesign -f -vvv -s "$application_cert" --entitlements Build/Mac/Resources/entitlements.plist --options runtime --timestamp  dist/Mac/BundleProject.app
```

Once again `codesign --verify --verbose dist/Mac/BundleProject.app` returns success. But this is not helpful, this tool just checks that a signature is present - not that the signature is valid nor the content of the app satisfies the notarization process.

Using `productbuild` we can recreate the package, and with `notarytool submit` to submit it for the notarization process. Finally, we can get a very satisfying...

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>id</key>
	<string>5a1ca8a8-bb3c-4772-a531-f4c79ae06ba1</string>
	<key>message</key>
	<string>Processing complete</string>
	<key>status</key>
	<string>Accepted</string>
</dict>
</plist>
```

Accepted 🎉🎉

## TL;DR

The code signing process integrated inside the Unreal Build Tool, using the modernized Xcode workflow, is not sufficient for notarizing and thus distributing an application. Moreover, if you are using CEF, you need to patch the engine or manually move the framework inside the bundle in the correct location. See [Chromium Embedded Framework as a macOS framework](#chromium-embedded-framework-as-a-macos-framework).

In any case, you need to:

1. Correctly configure the entitlements for your application. Be sure to have the bundle ID coherent in the `DefaultEngine.ini`, `Info.plist` and `entitlements.plist` files.
2. Have an `Info.plist` and update manually its bundle ID there. Unreal for some unknown reason is not doing it for you. Depending on your application, you may need to add other capabilities to the entitlements file.
3. Package the application passing the correct flags:
   ```sh
   LC_ALL="C" ue4 package Shipping -CrashReportClient
   ```
4. Ensure all the binaries provided by the engine (and also by the plugins) are executable (`$app` is where the `.app` is)
   ```sh
   find $app/Contents/UE/Engine/ -name '*dylib' -exec chmod +x {} \;
   ```
5. Manually codesign all the executables in the `.app`. Use the certificate starting with `Developer ID Application:`.
   ```sh
    to_sign=$(find dist/Mac/BundleProject.app/ -type f -perm +111 -print | grep -vE "\.(sh|bat)$" | \tr '\n' '\0')
    echo -n $to_sign | xargs -t -0 codesign -f -vvv -s "$application_cert" --entitlements Build/Mac/Resources/entitlements.plist --options runtime --timestamp
   ```
6. Codesign the application itself:
    ```sh
    codesign -f -vvv -s "$application_cert" --entitlements Build/Mac/Resources/entitlements.plist --options runtime --timestamp $app
    ```
7. Create the `.pkg` for notarization and future distribution.
   ```sh
    productbuild --component dist/Mac/BundleProject.app /Applications --sign "$installer_cert" --timestamp --identifier REPLACE_WITH_BUNDLE_ID Project.pkg
    ```
8. Submit the project to the notarization service and wait for the good feeling of seeing it as "Accepted".
    ```sh
    xcrun notarytool submit BundleProject.pkg --password $APPDEV_PASSWORD --apple-id $APPDEV_ID --team-id $APPDEV_TEAMID --wait --verbose --output-format plist
    ```

You can see a working example in the [dedicated repository](https://github.com:galeone/ue-bundle-project). Feel free to use it as a starting point for your macOS + Unreal Engine application.

## Conclusions

It has been a long journey. What has been written in this article is the result of weeks of trial and error, reading the poor close-to-zero documentation of the engine (there's documentation, but everything described in the article is not documented), and reading the source code of the Unreal Build Tool together with the documentation of the macOS distribution process for understanding how the various parts interact together.

The code signing integrated into the UBT is useless if you plan to distribute a valid application  - so an application that passes the notarization process. Moreover, all the various bugs found during this process made the experience really unpleasant. Anyway, I managed to solve/workaround all the various issues. So I hope this article can help other developers and save them all the weeks I had to spend on this.

A final note, the engine documentation about the modernized Xcode workflow focused on the creation of the `.xcarchive` - that is pretty much the same thing. The application created in the archive suffers from the very same issues.
