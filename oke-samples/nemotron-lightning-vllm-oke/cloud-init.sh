#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# OKE worker node cloud-init. Grow the root filesystem to the full boot volume, then run the
# standard OKE bootstrap. Without the first line, OKE node images keep a ~30 GB root partition
# regardless of the boot volume size and GPU nodes hit DiskPressure while pulling vLLM images.
set -eo pipefail
/usr/libexec/oci-growfs -y
curl --fail -H "Authorization: Bearer Oracle" -L0 http://169.254.169.254/opc/v2/instance/metadata/oke_init_script | base64 --decode >/var/run/oke-init.sh
bash /var/run/oke-init.sh
