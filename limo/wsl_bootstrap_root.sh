#!/usr/bin/env bash
# Run once as root in a freshly imported ubuntu-base WSL distro:
#   wsl -d <distro> -u root -- bash /mnt/c/.../limo/wsl_bootstrap_root.sh <user>
# Works for the ubuntu-base 20.04 and 24.04 images. Creates <user> with passwordless sudo (set a password later with `passwd`) and makes
# it the distro's default user.
set -euo pipefail
USER_NAME="${1:?usage: wsl_bootstrap_root.sh <user>}"
export DEBIAN_FRONTEND=noninteractive

# ubuntu-base is minimized; restore the basics a desktop distro would have
apt-get update
apt-get install -y --no-install-recommends sudo locales ca-certificates curl git gnupg \
  lsb-release software-properties-common bash-completion nano less tzdata
locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8

id "$USER_NAME" >/dev/null 2>&1 || useradd -m -s /bin/bash -G sudo "$USER_NAME"
echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/90-$USER_NAME"
chmod 440 "/etc/sudoers.d/90-$USER_NAME"
printf '[user]\ndefault=%s\n' "$USER_NAME" > /etc/wsl.conf
echo "bootstrap done; restart the distro (wsl --terminate <distro>) to log in as $USER_NAME"
