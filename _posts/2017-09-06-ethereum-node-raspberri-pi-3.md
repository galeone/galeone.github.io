---
layout: post
title: "Ethereum on Raspberry Pi: secure wallet and complete node with redundant storage"
date: 2017-09-06 13:00:00
summary: Ethereum is a relatively new player in the crypto-currencies ecosystem. If you are a researcher, an algorithmic trader or an investor, you could want to run an ethereum node to study, develop and store your ETH while contributing to the network good.
categories: raspberry ethereum archlinux
---

{:.center}
![Raspberry + Ethereum](/images/eth_raspy/eth_rasp.png)

In this post, I'm going to show you how to build a complete ethereum node and wallet using a raspberry pi 3. Why? Because running an ethereum node allows us to: contribute to the Ethereum network; store our wallets in a private and secure location; and it can be useful also when experimenting some kind of crypto-algorithmic trading (you can generate a new address from code, being sure that's synced and stored in a private location, do transactions and so on).

First of all, the requirements list (with Amazon's links!):

1. <a target="_blank" href="https://www.amazon.it/Raspberry-PI-Model-Scheda-madre/dp/B01CD5VC92/?&_encoding=UTF8&tag=pgale-21&linkCode=ur2&linkId=9e5bd1f82ff3c173d4d1846569c96c10&camp=3414&creative=21718">Raspberry Pi 3</a><img src="//ir-it.amazon-adsystem.com/e/ir?t=pgale-21&l=ur2&o=29" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" /> ~ 38 € / 45 $
2.  <a target="_blank" href="https://www.amazon.it/Zacro-Adattatore-Dissipatori-Raffreddamento-Ventilatore/dp/B06VWD8M6X?&_encoding=UTF8&tag=pgale-21&linkCode=ur2&linkId=c52ad9b92e28b72d21c933fb292bdf52&camp=3414&creative=21718">Case, power supply and heatsink kit</a><img src="//ir-it.amazon-adsystem.com/e/ir?t=pgale-21&l=ur2&o=29" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" /> ~ 14 € / 17 $
3. <a target="_blank" href="https://www.amazon.it/gp/product/B0089S2XJG/?&_encoding=UTF8&tag=pgale-21&linkCode=ur2&linkId=c4a046bf512b93cbd486bbe965a2d464&camp=3414&creative=21718">Very cheap UPS</a><img src="//ir-it.amazon-adsystem.com/e/ir?t=pgale-21&l=ur2&o=29" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" /> ~ 28 € / 34 $
4. <a target="_blank" href="https://www.amazon.it/gp/product/B00KWHJY7Q/?&_encoding=UTF8&tag=pgale-21&linkCode=ur2&linkId=6352a4eb356a3fc25a2a60bc5eab286c&camp=3414&creative=21718">2 x external HDD 1 TB</a><img src="//ir-it.amazon-adsystem.com/e/ir?t=pgale-21&l=ur2&o=29" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" /> ~ 2 x (52 € / 62 $) = 104 € / 124 $
5. <a target="_blank" href="https://www.amazon.it/gp/product/B01MRJNWLZ/?&_encoding=UTF8&tag=pgale-21&linkCode=ur2&linkId=94ea95a4b3a153fde6e7262b405a9a49&camp=3414&creative=21718">USB switch with external power supply</a><img src="//ir-it.amazon-adsystem.com/e/ir?t=pgale-21&l=ur2&o=29" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" /> ~ 16 € / 19 $
6. <a target="_blank" href="https://www.amazon.it/gp/product/B012PL7WKW/?&_encoding=UTF8&tag=pgale-21&linkCode=ur2&linkId=f2b388eecab1435b450079c0b614bd8f&camp=3414&creative=21718">32 GB Micro SD card</a><img src="//ir-it.amazon-adsystem.com/e/ir?t=pgale-21&l=ur2&o=29" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" /> ~ 17 € / 20 $

For a total expense of about 217 € / 259 $

A brief explanation of why we need that:

1. The Raspberry Pi 3 is the most supported ARM device around. It has 1 GB RAM, 4 USB 2.0 ports, and 1 ethernet port. It's possible to run Linux on it; in fact there are a lot of distributions available.
2. Keeping the Raspberry Pi powered on 24/7 produces heat: we have to dissipate it. This guarantees a long life to our device, plus in the kit we also find a nice plexiglass case that allows us to protect the device from external damaging.
3. A power outage is the second main cause of hardware failure after heating (or better, of filesystem corruption and subsequent data loss). To prevent serious damage a UPS is required. Also, the Raspberry Pi power consumption is not that high, therefore a UPS can keep it up for a reasonable amount of time (*hey reader, are you a Raspberry Pi hacker and you know how to use the GPIOs to monitor the wall outlet and trigger a safe shutdown procedure when the UPS is UP but the wall power is DOWN? Maybe you also know how to start it when to power comes back, and doing all this without any information from the UPS (no USB available)? Contact me!*).
4. The blockchain size increases in size day by day. A simple SD card is not enough to store the whole blockchain. Moreover, the SD card is a single point of failure and we should prevent that a failure makes us lose the blockchain and - most importantly - the wallet!
5. The Raspberry Pi does not provide enough power to the USB ports. Thus, since almost every external HDD is not externally powered, we have to provide enough power using a USB switch.
6. The Raspberry Pi comes with no storage, an SD card is required to install an operating system.

OK, let's start!

{% include inarticlead.html %}

## Archlinux ARM

I choose this one instead of the most common Raspbian because I love Archlinux. Also, it's really easy to install and use. More importantly, `geth` (the command line interface for running a full ethereum node) is already packaged and available in the community repository, therefore installing it is as easy as `pacman -Syu geth`.

Installing Archlinux ARM is straightforward. The [installation guide](https://archlinuxarm.org/platforms/armv8/broadcom/raspberry-pi-3) on how to create the SD card is pretty clear.
I'm going to show 2 ways to create the SD card. The first on Linux, the second on Windows.

### Prepare the SD card while running Linux

Insert the SD card into your SD card reader. The device `/dev/mmcblk0` appears.

```bash
# Become root:
su
# Create a new partition table
fdisk /dev/mmcblk0
# From the guide:
# At the fdisk prompt, delete old partitions and create a new one:
# Type o. This will clear out any partitions on the drive.
# Type p to list partitions. There should be no partitions left.
# Type n, then p for primary, 1 for the first partition on the drive, press ENTER to accept the default first sector, then type +100M for the last sector.
# Type t, then c to set the first partition to type W95 FAT32 (LBA).
# Type n, then p for primary, 2 for the second partition on the drive, and then press ENTER twice to accept the default first and last sector.
# Write the partition table and exit by typing w.

# Create a new work directory and move in
mkdir /tmp/wd
cd /tmp/wd

# Create and mount on /boot the FAT filesystem
mkfs.vfat /dev/mmcblk0p1
mkdir boot
mount /dev/mmcblk0p1 boot

# Create and mount on / the ext4 filesystem
mkfs.ext4 /dev/mmcblk0p2
mkdir root
mount /dev//mmcblk0p2 root

# Download and extract the root filesystem
wget http://os.archlinuxarm.org/os/ArchLinuxARM-rpi-2-latest.tar.gz

bsdtar -xpf ArchLinuxARM-rpi-2-latest.tar.gz -C root
sync

# Move boot files to the first partition:
mv root/boot/* boot

# Umount the two partitions
umount boot root
```

Now you can insert the SD card into the Raspberry Pi. Connect the UPS to the wall outlet, the Raspberry and the USB switch to the UPS, the HDDs to the switch and the switch to the Raspberry.
Connect the ethernet cable too and power it on.
Done! You can jump to the Archlinux configuration.

### Prepare the SD card while running Windows

If you're a Windows user, you can just:

1. Download Win32 Disk Imager: https://sourceforge.net/projects/win32diskimager/
2. Download the Archlinux ARM SD image: https://sourceforge.net/projects/archlinux-rpi2/
3. Run Win32 Disk Imager. Select the right device, the `.img` file downloaded and `Write`.

Your SD card is almost ready. Almost, because we have to resize the filesystem of the root partition, otherwise we only have 2GB available instead of the 32 GB that our SD card allows.
Now you can insert the SD card into the Raspberry Pi. Connect the UPS to the wall outlet, the Raspberry and the USB switch to the UPS, the HDDs to the switch and the switch to the Raspberry.
Connect the ethernet cable too and power it on.

Download [putty](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) and connect to the IP that the router assigned to the Raspberry Pi.
Username `alarm`, password `alarm`.

Now we have to resize the filesystem of the root partition, while the system is running. To do that we have to **delete** the partition (don't worry, data will be preserved) and create a new one that occupies all the free space.

```bash
# Become root
su # password root

# Use fdisk to change the partition table
fdisk /dev/mmcblk0
# At the fdisk prompt, delete the partition number 2
# Type d then 2. 
# Create a new primary partition (id 2)
# Type n then p. 
# At the prompt for the first sector and last sector just press enter.
# This will automatically start from the old first sector and will set the last
# sector to the end of the available space.
# Write the partition table and exit by typing w.

# Now reboot
reboot
```

Now log in again to the Raspberry Pi.

```bash
# become root
su # password root
# Now we use resize2fs to let the kernel aware that the filesystem has been enlarged
resize2fs /dev/mmcblk0p2
# logout
exit
```

Done. Now you're ready to configure your brand new Archlinux setup.

## Archlinux Configuration

Our aim is to setup a full ethereum node, whose data (blockchain and wallet) is stored onto the external hard drives. Before doing that we have to configure a RAID 1 for our external HDDs.

**Note**: the Archlinux image contains out-of-date software. This is **never** a good thing. Updated software give us better security. However, we're going to exploit this fact before upgrading the system.

The `ssh` server installed is so old that still allows, as a default option, the remote login of the `root` user. We're going to exploit this in order to create and mount the RAID 1 partition on `/home`. In this way, we can completely remove the home directory of the `alarm` user (and of any other user with a home directory) without any inconvenient.

Login remotely as root user (under windows use Putty with user `root` and password `root`, under Linux just user `ssh` with the same credentials)

### RAID setup

Our 2 external HDDs are located under `/dev/sda` and `/dev/sdb`. First of all, we're going to create a partition table on those devices and a partition for our data.

Since we're going to store a lot of data on these hard drives it's recommended to create a GPT partition table. For doing that, we need `gptfdisk`.

```bash
# Install gdisk
pacman -Sy gptfdisk

# Use it to partition both hard drives. Default values are OK since
# we want to create a single partition as big as the whole hard drives

gdisk /dev/sda
# Now create a single partition as big as the drive
# Type n then p.
# Accepts the defaults for every following question.
# Type w to write and exit

gdisk /dev/sdb
# Do the same exact procedure of above

# We created /dev/sda1 and /dev/sdb1 partitions
# Create an ext4 filesystem on these partitions
mkfs.ext4 /dev/sda1
mkfs.ext4 /dev/sdb1
```

Alright, we partitioned our HDDs. Now we can create a logical RAID 1 volume. RAID 1 is the most straightforward RAID level: straight mirroring. This means that if one of our drives fails, the block device provided by the RAID array will continue to function as normal.
We're now going to use `mdadm` to create the new block device `/dev/md0`:

```bash
mdadm --create --verbose --level=1 --metadata=1.2 --raid-devices=2 /dev/md0 /dev/sda1 /dev/sdb1
```

The last thing to do is to format the drive:

```bash
mkfs.ext4 /dev/md0
```

Done! Our drive is ready to use. (For a more comprehensive guide, please refer to the official documentation: https://wiki.archlinux.org/index.php/RAID)

### RAID /home and user setup

Since we're still logged as `root`, we can remove the current default user `alarm` and its home directory. Then we're going to mount our RAID device to `/home` in order to have redundancy for our data. Next, we create a new user to use `geth` with.

```bash
userdel -r alarm
mount /dev/md0 /home
# Add the user `geth`. Automatically creates /home/geth
useradd geth
# Set a new passowrd for user geth
passwd geth
```

Now, we need to make the mount of `/dev/md0` on `/home` permanent across reboots. Change the `/etc/fstab` adding the line (using `nano` or `vim`):

```fstab
/dev/md0        /home   ext4    defaults,nofail 0       0
```

Save the changes and exit.

Now that we have: a new user `geth`;  set a new password;  a RAID device that's mounted on `/home`;  we're ready to upgrade the whole system.

```bash
# First change the root password
passwd
# Now upgrade everything
pacman -Syu
# reboot
reboot
```

After the upgrade, we're unable to login via ssh as `root` (that's a good thing). Let's login as `geth`.

We're now going to install `geth`, configure it to start at boot and fix some little problem that using external HDDs can cause to our RAID-1 setup.

{% include inarticlead.html %}

### geth & startup services

Before installing `geth`, we have to disable any power saving options in our HDDs. We have already connected the external HDDs to an externally powered USB in order to give them enough power to work.
External hard drives slow down when the disk is not used and/or power them off. This can be a problem for our RAID configuration because once a drive turns off, it becomes unavailable as storage and so our RAID setup becomes useless.

We have to disable both HDDs power saving settings and we have to do it every time we turn on our Raspberry Pi.
A simple systemd service file is what we need.

```bash
su # become root
# install hdparm
pacman -S hdparm
```

Now, as root, use your preferred text editor and create the file `/etc/systemd/system/hdparm.service` with this content:

```systemd
[Unit]
Description=Disable HDDs power saving

[Service]
Type=oneshot
ExecStart=/usr/bin/hdparm -B 255 /dev/sda
ExecStart=/usr/bin/hdparm -B 255 /dev/sdb
ExecStart=/usr/bin/hdparm -S 0 /dev/sda
ExecStart=/usr/bin/hdparm -S 0 /dev/sdb

[Install]
WantedBy=multi-user.target
```

This service file will disable Advanced Power Management feature and the spin-down of both HDDs. Now we start the service and enable it on every boot.

```bash
systemctl start hdparm.service
systemctl enable hdparm.service
```

OK. We're finally ready to install `geth` and make it run on every boot.

```bash
# Install geth
pacman -S geth
```

Create the service file `/etc/systemd/system/geth@.service` with the following content:

```systemd
[Unit]
Description=geth Ethereum daemon
Requires=network.target

[Service]
Type=simple
User=%I
ExecStart=/usr/bin/geth --syncmode fast --cache 64 --maxpeers 12
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

You can change the `maxpeers` flag with the number of peers you desire. The highest your internet speed is, the highest can this number be (the default value is 25).

Let's enable this process on boot and make it run from the `geth` user.

```bash
systemctl enable geth@geth.service
systemctl start geth@geth.service
# back to geth user
exit
```

Done! Now the node is syncing with the ethereum network and you can use every feature of `geth` (here's the official doc: https://github.com/ethereum/go-ethereum/wiki/geth).

You can monitor the output of `geth` with `systemctl status geth@geth.service` and you can launch a console attached to the running node with `geth attach` (from the `geth` user).

**A small note**:

The blockchain synchronization can take a very long time. Also, `geth` uses a lot of RAM when syncing. Therefore is not unusual that the kernel kills the process and then systemd restart it. However, once the whole blockchain has been synchronized and the number of blocks to receive in an hour is low, a Raspberry Pi with 1 GB of RAM can handle the synchronization without crashing.

Ideally, if you already have a geth node synchronized, it's suggested to copy the blockchain using `scp` or any other remote copy tools instead of using `geth`.

## Conclusion

In this post, I described how to set up an Ethereum node on a Raspberry Pi 3 using Archlinux ARM. After the initial sync, we can use it to send/receive ETH, handle power failures thanks to the UPS and have redundant data to handle HDDs failures.

Please note that there are a lot of things to do to increase the security of your node. I just leave here few hints:

1. Do not expose the node outside the LAN
2. iptables with 'deny all' policy is a good friend
3. Configure your firewall in the router
4. Use strong passwords
5. Do not install `sudo`. If you install it, please add `Defaults rootpw` and use a different password for `geth` and `root`.
6. Place the node in a secure location
7. Evaluate if encryption of the RAID device is an option (although `geth` force us to protect our wallets with a passphrase, that's enough for my needs).

If you found this post helpful let me know it! Just leave a comment in the form below.

And since we are here to try ETH, if you want to offer me a beer (or anything else) you can just send me ETH to: `0xa1d283e77f8308559f62909526ccb2d9444d96fc`.
