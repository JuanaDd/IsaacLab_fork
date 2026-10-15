#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Install the fixed Newton source used by the H2 VBD task without changing other checkouts.
set -euo pipefail
lab_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
destination="$lab_root/.local/newton_h2_lab30"
repository="${NEWTON_REPOSITORY:-https://github.com/newton-physics/newton.git}"
revision="c2ca70bec5998062b0b2d35869865cbe3a49feee"

if [[ -e "$destination" ]]; then
    echo "Refusing to overwrite existing Newton checkout: $destination" >&2
    exit 1
fi

mkdir -p "$lab_root/.local"
git clone --no-checkout "$repository" "$destination"
git -C "$destination" checkout --detach "$revision"
git -C "$destination" apply --check "$lab_root/scripts/h2/newton_vbd.patch"
git -C "$destination" apply "$lab_root/scripts/h2/newton_vbd.patch"
echo "Prepared Newton 1.6.0 with the H2 sparse VBD patch. Run uv sync from the Isaac Lab root."
